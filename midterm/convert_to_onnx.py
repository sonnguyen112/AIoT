"""Export a checkpoint produced by ``train.py`` to ONNX.

Example::

    cd midterm
    python convert_to_onnx.py

The default input is ``checkpoints/best_resnet50.pth`` and the default output
is ``checkpoints/dodo_classifier.onnx``.  The exported model receives an
ImageNet-normalized float tensor with shape ``[batch, 3, 224, 224]`` and
returns two logits.  The class mapping and preprocessing values are stored in
the ONNX metadata when the ``onnx`` package is available.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import torch

try:
    from train import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, build_model
except ImportError:  # Allows: python -m midterm.convert_to_onnx
    from midterm.train import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, build_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Convert a trained ResNet50 PyTorch checkpoint to ONNX."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_dir / "checkpoints" / "best_resnet50.pth",
        help="PyTorch checkpoint produced by train.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "checkpoints" / "dodo_classifier.onnx",
        help="Destination ONNX file.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="Export device. CPU is the most portable default.",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy batch size used during export and optional verification.",
    )
    parser.add_argument(
        "--static-batch",
        action="store_true",
        help="Export a fixed batch dimension instead of a dynamic batch dimension.",
    )
    parser.add_argument(
        "--graph-optimization",
        "--optimization-level",
        dest="graph_optimization",
        choices=("disable", "basic", "extend", "extended", "layout", "all"),
        default="all",
        help=(
            "ONNX Runtime graph optimization level. "
            "'extend' is accepted as an alias for 'extended'."
        ),
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip ONNX checker and ONNX Runtime numerical verification.",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    """Resolve the requested export device."""
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load both current checkpoints and plain state-dict files."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # ``weights_only=False`` is needed by newer PyTorch versions because the
    # training checkpoint also contains argparse metadata and pathlib objects.
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Compatibility with older PyTorch versions.
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("The checkpoint must be a dictionary or a model state dict.")


def get_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Extract and normalize the model state dictionary."""
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid model_state_dict.")

    # Remove a DataParallel prefix if a checkpoint was trained on multiple GPUs.
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized[key.removeprefix("module.")] = value
    return normalized


def infer_num_classes(
    checkpoint: dict[str, Any], state_dict: dict[str, torch.Tensor]
) -> tuple[int, dict[str, int]]:
    """Read class information from the checkpoint, with a state-dict fallback."""
    class_to_idx = checkpoint.get("class_to_idx", {})
    if isinstance(class_to_idx, dict) and class_to_idx:
        class_to_idx = {str(name): int(index) for name, index in class_to_idx.items()}
        return len(class_to_idx), class_to_idx

    classifier_keys = ("fc.1.weight", "fc.weight")
    for key in classifier_keys:
        if key in state_dict:
            num_classes = int(state_dict[key].shape[0])
            return num_classes, {str(index): index for index in range(num_classes)}

    raise KeyError("Could not infer the number of classes from the checkpoint.")


def optimization_level(name: str) -> Any:
    """Map a CLI name to the ONNX Runtime graph optimization enum."""
    import onnxruntime as ort

    normalized_name = "extended" if name == "extend" else name
    levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    if normalized_name == "layout":
        layout_level = getattr(ort.GraphOptimizationLevel, "ORT_ENABLE_LAYOUT", None)
        if layout_level is None:
            raise RuntimeError(
                "Phiên bản onnxruntime hiện tại không hỗ trợ mức 'layout'."
            )
        return layout_level
    return levels[normalized_name]


def optimize_onnx(
    raw_path: Path,
    output_path: Path,
    graph_optimization: str,
) -> None:
    """Serialize an ONNX Runtime-optimized graph to ``output_path``."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "Cần onnxruntime để áp dụng graph optimization. "
            "Hãy chạy: pip install onnxruntime"
        ) from exc

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = optimization_level(graph_optimization)
    session_options.optimized_model_filepath = str(output_path)
    output_path.unlink(missing_ok=True)
    # Creating the session performs the optimization and serializes the result.
    ort.InferenceSession(
        str(raw_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


def add_metadata(
    output_path: Path,
    class_to_idx: dict[str, int],
    image_size: int,
    graph_optimization: str,
) -> None:
    """Validate the ONNX graph and attach preprocessing metadata when possible."""
    try:
        import onnx
    except ImportError:
        print("onnx is not installed; skipped graph checking and metadata.")
        print("Install it with: pip install onnx")
        return

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    metadata = {
        "class_to_idx": json.dumps(class_to_idx, sort_keys=True),
        "image_size": str(image_size),
        "input_layout": "NCHW",
        "input_dtype": "float32",
        "normalization_mean": json.dumps(IMAGENET_MEAN),
        "normalization_std": json.dumps(IMAGENET_STD),
        "graph_optimization": "extended" if graph_optimization == "extend" else graph_optimization,
    }
    for key, value in metadata.items():
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(onnx_model, str(output_path))
    print("ONNX graph check: OK")


def verify_with_onnxruntime(
    output_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    image_size: int,
) -> None:
    """Compare ONNX Runtime output with PyTorch output when available."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime is not installed; skipped numerical verification.")
        print("Install it with: pip install onnxruntime")
        return

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    test_input = torch.randn(batch_size, 3, image_size, image_size, device=device)
    with torch.inference_mode():
        torch_output = model(test_input).detach().cpu().numpy()
    onnx_output = session.run(None, {input_name: test_input.cpu().numpy()})[0]
    max_difference = float(np.max(np.abs(torch_output - onnx_output)))
    print(f"ONNX Runtime max absolute difference: {max_difference:.6g}")
    if max_difference > 1e-4:
        raise RuntimeError("ONNX Runtime output differs from PyTorch by more than 1e-4.")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.opset < 11:
        raise ValueError("--opset must be at least 11.")

    device = choose_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    state_dict = get_state_dict(checkpoint)
    num_classes, class_to_idx = infer_num_classes(checkpoint, state_dict)
    model = build_model(num_classes=num_classes, use_pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    image_size = int(checkpoint.get("image_size", IMAGE_SIZE))
    dummy_input = torch.randn(args.batch_size, 3, image_size, image_size, device=device)
    dynamic_axes = None
    if not args.static_batch:
        dynamic_axes = {"images": {0: "batch"}, "logits": {0: "batch"}}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs: dict[str, Any] = {
        "input_names": ["images"],
        "output_names": ["logits"],
        "dynamic_axes": dynamic_axes,
        "opset_version": args.opset,
        "do_constant_folding": True,
    }
    with tempfile.TemporaryDirectory(prefix="onnx_raw_", dir=args.output.parent) as temp_dir:
        raw_path = Path(temp_dir) / "raw.onnx"
        try:
            torch.onnx.export(model, dummy_input, str(raw_path), dynamo=False, **export_kwargs)
        except TypeError:  # Compatibility with older PyTorch versions.
            torch.onnx.export(model, dummy_input, str(raw_path), **export_kwargs)
        optimize_onnx(raw_path, args.output, args.graph_optimization)

    print(f"Exported: {args.output}")
    print(f"Device: {device} | Input: [batch, 3, {image_size}, {image_size}]")
    print(f"Classes: {class_to_idx}")
    print(f"Graph optimization: {args.graph_optimization}")

    if not args.skip_check:
        add_metadata(args.output, class_to_idx, image_size, args.graph_optimization)
        verify_with_onnxruntime(args.output, model, device, args.batch_size, image_size)


if __name__ == "__main__":
    main()
