"""Benchmark PyTorch CPU against the three exported ONNX graphs.

The default run repeats the complete ``dataset/test`` set 1,000 times for
each model.  Preprocessing is performed once before timing, so the reported
latency measures model inference rather than JPEG decoding or resizing.

Each model runs in a fresh child process.  This keeps RSS/CPU measurements
independent from the framework and allocator state left by another model.

Example::

    cd /home/sonnguyen112/data/Master/AIoT/project
    python midterm/benchmark.py --iterations 1000 --threads 1

Outputs are written to ``midterm/benchmark_results`` by default:

* ``benchmark_summary.csv``: one row per model, convenient for spreadsheets;
* ``benchmark_per_image.csv``: first-pass prediction and accuracy per image;
* ``benchmark_report.json``: full latency percentiles, resource metrics,
  environment details and ONNX graph/node-fusion analysis.

The three ONNX files are loaded with ``ORT_DISABLE_ALL`` by default.  Their
serialized graphs therefore remain the objects being compared.  Use
``--onnx-runtime-optimization all`` when measuring runtime re-optimization as
well.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
import queue
import resource
import sys
import threading
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_SIZE = 224
PREPROCESS_SIZE = 256
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_ORDER = ("pth_cpu", "onnx_basic", "onnx_extend", "onnx_all")


def parse_args() -> argparse.Namespace:
    """Parse benchmark options."""
    project_dir = Path(__file__).resolve().parent
    checkpoints_dir = project_dir / "checkpoints"
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ResNet50 PyTorch CPU and three ONNX Runtime CPU graphs "
            "on the complete test set."
        )
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=project_dir / "dataset" / "test",
        help="Test folder containing one directory per class.",
    )
    parser.add_argument(
        "--pth",
        type=Path,
        default=checkpoints_dir / "best_resnet50.pth",
        help="PyTorch checkpoint to benchmark on CPU.",
    )
    parser.add_argument(
        "--onnx-basic",
        type=Path,
        default=checkpoints_dir / "dodo_classifier_basic.onnx",
        help="ONNX graph exported with ORT basic optimization.",
    )
    parser.add_argument(
        "--onnx-extend",
        type=Path,
        default=checkpoints_dir / "dodo_classifier_extend.onnx",
        help="ONNX graph exported with ORT extended optimization.",
    )
    parser.add_argument(
        "--onnx-all",
        type=Path,
        default=checkpoints_dir / "dodo_classifier.onnx",
        help="ONNX graph exported with ORT all optimization.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of complete test-set repetitions per model.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Complete test-set repetitions excluded from timing.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Same intra-op CPU thread count for PyTorch and ONNX Runtime.",
    )
    parser.add_argument(
        "--memory-sample-ms",
        type=float,
        default=50.0,
        help="RSS sampling period used for peak RAM measurement.",
    )
    parser.add_argument(
        "--onnx-runtime-optimization",
        choices=("disable", "basic", "extended", "all"),
        default="disable",
        help=(
            "Optimization applied again when loading ONNX. The default keeps "
            "the serialized graph unchanged."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "benchmark_results",
        help="Folder for CSV and JSON reports.",
    )
    return parser.parse_args()


def package_version(distribution: str) -> str | None:
    """Return an installed distribution version without importing the package."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_report(threads: int, runtime_optimization: str) -> dict[str, Any]:
    """Collect the hardware/software context needed to reproduce the run."""
    cpu_model = None
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.is_file():
        try:
            for line in cpuinfo_path.read_text(errors="replace").splitlines():
                if line.lower().startswith("model name") or line.lower().startswith("hardware"):
                    cpu_model = line.split(":", 1)[-1].strip()
                    break
        except OSError:
            pass

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count() or 1,
        "threads_per_runtime": threads,
        "pytorch_version": package_version("torch"),
        "torchvision_version": package_version("torchvision"),
        "onnxruntime_version": package_version("onnxruntime"),
        "onnx_version": package_version("onnx"),
        "numpy_version": package_version("numpy"),
        "pillow_version": package_version("Pillow"),
        "onnx_execution_provider": "CPUExecutionProvider",
        "onnx_runtime_graph_optimization_on_load": runtime_optimization,
    }


def discover_test_images(test_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Discover labeled images and reproduce ImageFolder's sorted class order."""
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    class_dirs = sorted(path for path in test_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class directories found in {test_dir}")
    class_to_idx = {path.name: index for index, path in enumerate(class_dirs)}
    records: list[dict[str, Any]] = []
    for class_dir in class_dirs:
        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image_path in image_paths:
            records.append(
                {
                    "path": str(image_path.resolve()),
                    "label": class_dir.name,
                    "label_index": class_to_idx[class_dir.name],
                }
            )
    if not records:
        raise ValueError(f"No supported images found in {test_dir}")
    return records, class_to_idx


def preprocess_image(path: str | Path) -> np.ndarray:
    """Apply the same deterministic preprocessing used by training/demo.py."""
    with Image.open(path) as source:
        image = source.convert("RGB")
    scale = PREPROCESS_SIZE / min(image.width, image.height)
    resized_width = max(IMAGE_SIZE, round(image.width * scale))
    resized_height = max(IMAGE_SIZE, round(image.height * scale))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    image = image.resize((resized_width, resized_height), resampling)
    left = (resized_width - IMAGE_SIZE) // 2
    top = (resized_height - IMAGE_SIZE) // 2
    image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    # copy() gives torch.from_numpy a writable, contiguous array.
    return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32, copy=True)


def load_inputs(records: list[dict[str, Any]]) -> list[np.ndarray]:
    """Preprocess every test image once before model timing starts."""
    return [preprocess_image(record["path"]) for record in records]


def read_rss_bytes() -> int:
    """Read current resident memory without requiring psutil."""
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        try:
            for line in status_path.read_text(errors="replace").splitlines():
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
        except (OSError, ValueError):
            pass

    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # Linux reports KiB; macOS reports bytes.
        return int(usage.ru_maxrss * (1024 if sys.platform != "darwin" else 1))
    except (AttributeError, OSError):
        return 0


class MemorySampler:
    """Sample process RSS during inference and retain the observed peak."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = max(interval_seconds, 0.001)
        self.stop_event = threading.Event()
        self.peak_bytes = read_rss_bytes()
        self.thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            self.peak_bytes = max(self.peak_bytes, read_rss_bytes())

    def start(self) -> None:
        self.thread = threading.Thread(target=self._sample, name="rss-sampler", daemon=True)
        self.thread.start()

    def stop(self) -> int:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self.peak_bytes = max(self.peak_bytes, read_rss_bytes())
        return self.peak_bytes


def parse_onnx_runtime_optimization(ort: Any, name: str) -> Any:
    """Map the CLI runtime optimization name to ONNX Runtime's enum."""
    return {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }[name]


def load_pytorch_model(path: Path, threads: int) -> tuple[Any, Any]:
    """Load the checkpoint and configure PyTorch for CPU inference."""
    import torch

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Some PyTorch builds initialize the inter-op pool during import.
        pass

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("The PyTorch checkpoint must contain a dictionary.")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("The checkpoint does not contain a valid model_state_dict.")
    state_dict = {
        str(key).removeprefix("module."): value for key, value in state_dict.items()
    }
    class_to_idx = checkpoint.get("class_to_idx")
    if not isinstance(class_to_idx, dict) or not class_to_idx:
        classifier_key = "fc.1.weight" if "fc.1.weight" in state_dict else "fc.weight"
        class_count = int(state_dict[classifier_key].shape[0])
        class_to_idx = {str(index): index for index in range(class_count)}

    try:
        from train import build_model
    except ImportError:
        from midterm.train import build_model

    model = build_model(num_classes=len(class_to_idx), use_pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    model.to("cpu").eval()
    return torch, model


def load_onnx_model(
    path: Path,
    threads: int,
    runtime_optimization: str,
) -> tuple[Any, Any, str]:
    """Load an ONNX model with a controlled CPU Runtime session."""
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = parse_onnx_runtime_optimization(
        ort, runtime_optimization
    )
    session_options.intra_op_num_threads = threads
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    return ort, session, input_name


def softmax_confidence(logits: np.ndarray) -> float:
    """Return the largest softmax probability for one output batch."""
    values = np.asarray(logits, dtype=np.float32)
    values = values - np.max(values, axis=1, keepdims=True)
    probabilities = np.exp(values)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return float(np.max(probabilities[0]))


def summarize_latencies(latency_ns: list[int]) -> dict[str, float]:
    """Calculate latency statistics in milliseconds."""
    values = np.asarray(latency_ns, dtype=np.float64) / 1_000_000.0
    return {
        "inference_time_mean_ms": float(np.mean(values)),
        "inference_time_median_ms": float(np.median(values)),
        "inference_time_std_ms": float(np.std(values)),
        "inference_time_min_ms": float(np.min(values)),
        "inference_time_max_ms": float(np.max(values)),
        "inference_time_p50_ms": float(np.percentile(values, 50)),
        "inference_time_p90_ms": float(np.percentile(values, 90)),
        "inference_time_p95_ms": float(np.percentile(values, 95)),
        "inference_time_p99_ms": float(np.percentile(values, 99)),
        "inference_time_total_s": float(np.sum(values) / 1000.0),
    }


def benchmark_model(
    model_key: str,
    weight_path: Path,
    input_arrays: list[np.ndarray],
    records: list[dict[str, Any]],
    iterations: int,
    warmup: int,
    threads: int,
    memory_sample_ms: float,
    onnx_runtime_optimization: str,
) -> dict[str, Any]:
    """Load and benchmark one model inside its dedicated child process."""
    rss_before_load = read_rss_bytes()
    load_start = time.perf_counter()

    if model_key == "pth_cpu":
        torch, model = load_pytorch_model(weight_path, threads)
        torch_inputs = [torch.from_numpy(array) for array in input_arrays]
        runtime_name = "PyTorch CPU"
    else:
        _, session, input_name = load_onnx_model(
            weight_path, threads, onnx_runtime_optimization
        )
        torch = None
        model = session
        torch_inputs = input_arrays
        runtime_name = "ONNX Runtime CPUExecutionProvider"

    load_time_ms = (time.perf_counter() - load_start) * 1000.0
    rss_after_load = read_rss_bytes()

    def run_once(input_value: Any) -> Any:
        if model_key == "pth_cpu":
            return model(input_value)
        return model.run(None, {input_name: input_value})[0]

    # Warm up the exact same input sequence used for timed inference.
    if model_key == "pth_cpu":
        with torch.inference_mode():
            for _ in range(warmup):
                for input_value in torch_inputs:
                    run_once(input_value)
    else:
        for _ in range(warmup):
            for input_value in torch_inputs:
                run_once(input_value)

    gc.collect()
    latency_ns: list[int] = []
    per_image: list[dict[str, Any]] = []
    correct = 0
    total = iterations * len(records)
    memory_sampler = MemorySampler(memory_sample_ms / 1000.0)

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    memory_sampler.start()

    if model_key == "pth_cpu":
        with torch.inference_mode():
            for iteration in range(iterations):
                for index, input_value in enumerate(torch_inputs):
                    start_ns = time.perf_counter_ns()
                    logits = run_once(input_value)
                    elapsed_ns = time.perf_counter_ns() - start_ns
                    latency_ns.append(elapsed_ns)
                    prediction = int(logits.argmax(dim=1).item())
                    expected = int(records[index]["label_index"])
                    correct += int(prediction == expected)
                    if iteration == 0:
                        per_image.append(
                            {
                                "image": records[index]["path"],
                                "true_label": records[index]["label"],
                                "true_index": expected,
                                "predicted_index": prediction,
                                "predicted_label": prediction_label(records, prediction),
                                "correct": prediction == expected,
                                "confidence": softmax_confidence(
                                    logits.detach().cpu().numpy()
                                ),
                            }
                        )
    else:
        for iteration in range(iterations):
            for index, input_value in enumerate(input_arrays):
                start_ns = time.perf_counter_ns()
                logits = run_once(input_value)
                elapsed_ns = time.perf_counter_ns() - start_ns
                latency_ns.append(elapsed_ns)
                prediction = int(np.argmax(logits, axis=1)[0])
                expected = int(records[index]["label_index"])
                correct += int(prediction == expected)
                if iteration == 0:
                    per_image.append(
                        {
                            "image": records[index]["path"],
                            "true_label": records[index]["label"],
                            "true_index": expected,
                            "predicted_index": prediction,
                            "predicted_label": prediction_label(records, prediction),
                            "correct": prediction == expected,
                            "confidence": softmax_confidence(logits),
                        }
                    )

    wall_elapsed_s = time.perf_counter() - wall_start
    cpu_elapsed_s = time.process_time() - cpu_start
    peak_rss = memory_sampler.stop()

    latency_stats = summarize_latencies(latency_ns)
    measured_inference_s = latency_stats["inference_time_total_s"]
    average_input_bytes = float(np.mean([array.nbytes for array in input_arrays]))
    measured_throughput = len(latency_ns) / max(measured_inference_s, np.finfo(float).eps)
    wall_throughput = len(latency_ns) / max(wall_elapsed_s, np.finfo(float).eps)
    logical_cpu_count = os.cpu_count() or 1

    return {
        "model_key": model_key,
        "runtime": runtime_name,
        "weight": str(weight_path.resolve()),
        "model_size_bytes": weight_path.stat().st_size,
        "model_size_mib": weight_path.stat().st_size / (1024.0**2),
        "flash_bytes": weight_path.stat().st_size,
        "flash_mib": weight_path.stat().st_size / (1024.0**2),
        "load_time_ms": load_time_ms,
        "rss_before_load_mib": rss_before_load / (1024.0**2),
        "rss_after_load_mib": rss_after_load / (1024.0**2),
        "rss_delta_after_load_mib": (rss_after_load - rss_before_load) / (1024.0**2),
        "peak_rss_during_inference_mib": peak_rss / (1024.0**2),
        "peak_rss_delta_after_load_mib": (peak_rss - rss_after_load) / (1024.0**2),
        "cpu_time_inference_s": cpu_elapsed_s,
        "cpu_percent_one_core": 100.0 * cpu_elapsed_s / max(wall_elapsed_s, 1e-12),
        "cpu_percent_all_cores": (
            100.0 * cpu_elapsed_s / max(wall_elapsed_s, 1e-12) / logical_cpu_count
        ),
        "wall_time_inference_s": wall_elapsed_s,
        "total_inferences": len(latency_ns),
        "iterations": iterations,
        "test_images": len(records),
        "accuracy": correct / total,
        "accuracy_percent": 100.0 * correct / total,
        "correct_predictions": correct,
        "average_input_bytes": average_input_bytes,
        "input_mib_processed": len(latency_ns) * average_input_bytes / (1024.0**2),
        "throughput_images_per_second": measured_throughput,
        "throughput_input_mib_per_second": (
            measured_throughput * average_input_bytes / (1024.0**2)
        ),
        "wall_throughput_images_per_second": wall_throughput,
        "per_image": per_image,
        **latency_stats,
    }


def prediction_label(records: list[dict[str, Any]], index: int) -> str:
    """Resolve a prediction index using the sorted test class folders."""
    labels = sorted({str(record["label"]) for record in records})
    return labels[index] if 0 <= index < len(labels) else f"class_{index}"


def graph_analysis(path: Path, reference_node_count: int | None) -> dict[str, Any]:
    """Inspect serialized ONNX nodes and quantify reductions/fused operators."""
    try:
        import onnx
    except ImportError as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    model = onnx.load(str(path))
    nodes = list(model.graph.node)
    op_counts = Counter(node.op_type for node in nodes)
    fused_nodes = [
        node
        for node in nodes
        if "fused" in node.op_type.lower() or "fused" in node.name.lower()
    ]
    layout_nodes = [
        node
        for node in nodes
        if "_nchwc" in node.name.lower() or node.op_type.lower() == "reorderoutput"
    ]
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    node_reduction = None
    node_reduction_percent = None
    if reference_node_count is not None:
        node_reduction = reference_node_count - len(nodes)
        node_reduction_percent = 100.0 * node_reduction / reference_node_count

    return {
        "available": True,
        "graph_optimization": metadata.get("graph_optimization"),
        "node_count": len(nodes),
        "reference_basic_node_count": reference_node_count,
        "nodes_removed_vs_basic": node_reduction,
        "node_reduction_percent_vs_basic": node_reduction_percent,
        "fused_node_count": len(fused_nodes),
        "fused_node_names": [node.name for node in fused_nodes],
        "fused_op_type_counts": dict(Counter(node.op_type for node in fused_nodes)),
        "layout_node_count": len(layout_nodes),
        "layout_op_type_counts": dict(Counter(node.op_type for node in layout_nodes)),
        "op_type_counts": dict(op_counts),
    }


def worker_entry(
    model_key: str,
    weight_path: str,
    input_arrays: list[np.ndarray],
    records: list[dict[str, Any]],
    iterations: int,
    warmup: int,
    threads: int,
    memory_sample_ms: float,
    onnx_runtime_optimization: str,
    result_queue: Any,
) -> None:
    """Multiprocessing entry point that serializes success/errors to a queue."""
    try:
        result = benchmark_model(
            model_key=model_key,
            weight_path=Path(weight_path),
            input_arrays=input_arrays,
            records=records,
            iterations=iterations,
            warmup=warmup,
            threads=threads,
            memory_sample_ms=memory_sample_ms,
            onnx_runtime_optimization=onnx_runtime_optimization,
        )
        result_queue.put({"ok": True, "result": result})
    except BaseException as exc:  # Return traceback to the parent process.
        result_queue.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def run_in_child(
    model_key: str,
    weight_path: Path,
    input_arrays: list[np.ndarray],
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run one model in an isolated spawn child and return its report."""
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=worker_entry,
        args=(
            model_key,
            str(weight_path),
            input_arrays,
            records,
            args.iterations,
            args.warmup,
            args.threads,
            args.memory_sample_ms,
            args.onnx_runtime_optimization,
            result_queue,
        ),
    )
    process.start()
    process.join()
    try:
        message = result_queue.get_nowait()
    except queue.Empty as exc:
        raise RuntimeError(
            f"Benchmark child for {model_key} exited with code {process.exitcode} "
            "without returning a report."
        ) from exc
    finally:
        result_queue.close()
        result_queue.join_thread()

    if not message.get("ok"):
        raise RuntimeError(
            f"Benchmark failed for {model_key}: {message.get('error')}\n"
            f"{message.get('traceback', '')}"
        )
    return message["result"]


def flatten_summary_row(result: dict[str, Any]) -> dict[str, Any]:
    """Select scalar fields for the spreadsheet-friendly summary CSV."""
    graph = result.get("graph", {})
    fields = [
        "model_key",
        "runtime",
        "weight",
        "model_size_bytes",
        "model_size_mib",
        "flash_bytes",
        "flash_mib",
        "load_time_ms",
        "rss_before_load_mib",
        "rss_after_load_mib",
        "rss_delta_after_load_mib",
        "peak_rss_during_inference_mib",
        "peak_rss_delta_after_load_mib",
        "cpu_time_inference_s",
        "cpu_percent_one_core",
        "cpu_percent_all_cores",
        "wall_time_inference_s",
        "total_inferences",
        "iterations",
        "test_images",
        "accuracy",
        "accuracy_percent",
        "correct_predictions",
        "inference_time_mean_ms",
        "inference_time_median_ms",
        "inference_time_std_ms",
        "inference_time_min_ms",
        "inference_time_max_ms",
        "inference_time_p50_ms",
        "inference_time_p90_ms",
        "inference_time_p95_ms",
        "inference_time_p99_ms",
        "inference_time_total_s",
        "average_input_bytes",
        "input_mib_processed",
        "throughput_images_per_second",
        "throughput_input_mib_per_second",
        "wall_throughput_images_per_second",
        "speedup_vs_pth",
        "latency_delta_vs_pth_ms",
    ]
    row = {field: result.get(field) for field in fields}
    row.update(
        {
            "graph_optimization": graph.get("graph_optimization"),
            "graph_node_count": graph.get("node_count"),
            "reference_basic_node_count": graph.get("reference_basic_node_count"),
            "nodes_removed_vs_basic": graph.get("nodes_removed_vs_basic"),
            "node_reduction_percent_vs_basic": graph.get(
                "node_reduction_percent_vs_basic"
            ),
            "fused_node_count": graph.get("fused_node_count"),
            "fused_op_type_counts": json.dumps(
                graph.get("fused_op_type_counts", {}), sort_keys=True
            ),
            "layout_node_count": graph.get("layout_node_count"),
            "op_type_counts": json.dumps(graph.get("op_type_counts", {}), sort_keys=True),
        }
    )
    return row


def write_reports(
    output_dir: Path,
    report: dict[str, Any],
    results: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    """Write JSON, summary CSV and per-image CSV reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark_report.json"
    summary_path = output_dir / "benchmark_summary.csv"
    per_image_path = output_dir / "benchmark_per_image.csv"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_rows = [flatten_summary_row(result) for result in results]
    summary_fields = list(summary_rows[0])
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    per_image_fields = [
        "model_key",
        "runtime",
        "image",
        "true_label",
        "true_index",
        "predicted_label",
        "predicted_index",
        "correct",
        "confidence",
    ]
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_image_fields)
        writer.writeheader()
        for result in results:
            for item in result["per_image"]:
                writer.writerow(
                    {
                        "model_key": result["model_key"],
                        "runtime": result["runtime"],
                        **{field: item.get(field) for field in per_image_fields[2:]},
                    }
                )
    return json_path, summary_path, per_image_path


def main() -> None:
    """Run all model benchmarks and write the report files."""
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1.")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative.")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1.")
    if args.memory_sample_ms <= 0:
        raise ValueError("--memory-sample-ms must be positive.")

    model_paths = {
        "pth_cpu": args.pth,
        "onnx_basic": args.onnx_basic,
        "onnx_extend": args.onnx_extend,
        "onnx_all": args.onnx_all,
    }
    for path in model_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")

    records, class_to_idx = discover_test_images(args.test_dir)
    input_arrays = load_inputs(records)
    input_bytes = int(input_arrays[0].nbytes)
    print(
        f"Dataset: {len(records)} images | repetitions: {args.iterations} | "
        f"timed inferences/model: {len(records) * args.iterations} | "
        f"input: {input_bytes} bytes/image",
        flush=True,
    )

    graph_reference_path = model_paths["onnx_basic"]
    try:
        import onnx

        basic_node_count = len(onnx.load(str(graph_reference_path)).graph.node)
    except ImportError:
        basic_node_count = None

    results: list[dict[str, Any]] = []
    for model_key in MODEL_ORDER:
        print(f"Benchmarking {model_key}: {model_paths[model_key]}", flush=True)
        result = run_in_child(
            model_key=model_key,
            weight_path=model_paths[model_key],
            input_arrays=input_arrays,
            records=records,
            args=args,
        )
        if model_key == "pth_cpu":
            result["graph"] = {
                "available": False,
                "reason": "PyTorch checkpoint has no serialized ONNX graph.",
            }
        else:
            result["graph"] = graph_analysis(model_paths[model_key], basic_node_count)
        results.append(result)
        print(
            f"  mean={result['inference_time_mean_ms']:.4f} ms | "
            f"accuracy={result['accuracy_percent']:.2f}% | "
            f"throughput={result['throughput_images_per_second']:.2f} images/s",
            flush=True,
        )

    pth_mean = next(result["inference_time_mean_ms"] for result in results if result["model_key"] == "pth_cpu")
    for result in results:
        result["speedup_vs_pth"] = pth_mean / result["inference_time_mean_ms"]
        result["latency_delta_vs_pth_ms"] = (
            result["inference_time_mean_ms"] - pth_mean
        )

    report = {
        "benchmark": {
            "description": "Repeated inference benchmark on the complete test set.",
            "iterations_are_complete_test_set_repetitions": True,
            "preprocessing_in_timing": False,
            "warmup_repetitions": args.warmup,
            "iterations_per_model": args.iterations,
            "test_image_count": len(records),
            "timed_inferences_per_model": len(records) * args.iterations,
            "class_to_idx": class_to_idx,
            "test_dir": str(args.test_dir.resolve()),
            "threads_per_runtime": args.threads,
            "onnx_runtime_optimization_on_load": args.onnx_runtime_optimization,
            "node_fusion_reference": "onnx_basic graph node count",
        },
        "environment": environment_report(args.threads, args.onnx_runtime_optimization),
        "models": results,
    }
    json_path, summary_path, per_image_path = write_reports(args.output_dir, report, results)
    print(f"JSON report: {json_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Per-image CSV: {per_image_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
