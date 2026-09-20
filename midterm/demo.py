"""Small Gradio web UI for testing the dodo classifier.

Run from this directory with::

    pip install torch torchvision pillow numpy gradio onnxruntime
    python demo.py

The UI accepts a local ``.pth`` or ``.onnx`` weight from the checkpoints
folder, or lets the user upload one.  It then accepts an image and displays
the predicted class and both class probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_SIZE = 224
PREPROCESS_SIZE = 256
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
DEFAULT_CLASS_TO_IDX = {"dodo": 0, "no_dodo": 1}


def parse_args() -> argparse.Namespace:
    """Parse server options."""
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Web demo for the dodo classifier.")
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=project_dir / "checkpoints",
        help="Folder scanned for local .pth and .onnx files.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used for .pth inference. ONNX inference uses CPU.",
    )
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share URL.")
    return parser.parse_args()


def normalize_class_to_idx(value: Any) -> dict[str, int]:
    """Convert checkpoint/ONNX metadata into a predictable class mapping."""
    if isinstance(value, dict) and value:
        return {str(name): int(index) for name, index in value.items()}
    return DEFAULT_CLASS_TO_IDX.copy()


def names_by_index(class_to_idx: dict[str, int]) -> dict[int, str]:
    """Invert a class-to-index mapping."""
    return {index: name for name, index in class_to_idx.items()}


def preprocess_image(image: Image.Image, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Match the validation preprocessing used during training."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image = image.convert("RGB")

    if image.width < 1 or image.height < 1:
        raise ValueError("Ảnh không có kích thước hợp lệ.")

    scale = PREPROCESS_SIZE / min(image.width, image.height)
    resized_width = max(image_size, round(image.width * scale))
    resized_height = max(image_size, round(image.height * scale))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    image = image.resize((resized_width, resized_height), resampling)

    left = (resized_width - image_size) // 2
    top = (resized_height - image_size) // 2
    image = image.crop((left, top, left + image_size, top + image_size))

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute a numerically stable softmax for one output batch."""
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / np.sum(probabilities, axis=1, keepdims=True)


class InferenceEngine:
    """Load and cache either PyTorch or ONNX models selected by the user."""

    def __init__(self, device_request: str = "auto") -> None:
        self.device_request = device_request
        self.pth_cache: dict[tuple[str, int, str], tuple[Any, dict[str, int], int, Any]] = {}
        self.onnx_cache: dict[tuple[str, int], tuple[Any, dict[str, int], int, str]] = {}

    def _torch(self) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Chưa cài PyTorch. Hãy chạy: pip install torch torchvision"
            ) from exc
        return torch

    def _torch_device(self, torch: Any) -> Any:
        if self.device_request == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Đã chọn CUDA nhưng CUDA không khả dụng.")
            return torch.device("cuda")
        if self.device_request == "auto" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def _load_checkpoint(torch: Any, path: Path, device: Any) -> dict[str, Any]:
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)
        if not isinstance(checkpoint, dict):
            raise TypeError("Checkpoint .pth phải là một dictionary hoặc state_dict.")
        return checkpoint

    def _load_pth(self, path: Path) -> tuple[Any, dict[str, int], int, Any]:
        torch = self._torch()
        device = self._torch_device(torch)
        stat = path.stat()
        cache_key = (str(path.resolve()), stat.st_mtime_ns, str(device))
        if cache_key in self.pth_cache:
            return self.pth_cache[cache_key]

        checkpoint = self._load_checkpoint(torch, path, device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise TypeError("Checkpoint không chứa model_state_dict hợp lệ.")
        state_dict = {
            str(key).removeprefix("module."): value for key, value in state_dict.items()
        }

        class_to_idx = normalize_class_to_idx(checkpoint.get("class_to_idx"))
        image_size = int(checkpoint.get("image_size", IMAGE_SIZE))
        try:
            from train import build_model
        except ImportError:
            from midterm.train import build_model

        model = build_model(num_classes=len(class_to_idx), use_pretrained=False)
        model.load_state_dict(state_dict, strict=True)
        model.to(device).eval()
        loaded = (model, class_to_idx, image_size, device)
        self.pth_cache[cache_key] = loaded
        return loaded

    def _load_onnx(self, path: Path) -> tuple[Any, dict[str, int], int, str]:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "Chưa cài ONNX Runtime. Hãy chạy: pip install onnxruntime"
            ) from exc

        stat = path.stat()
        cache_key = (str(path.resolve()), stat.st_mtime_ns)
        if cache_key in self.onnx_cache:
            return self.onnx_cache[cache_key]

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        metadata = session.get_modelmeta().custom_metadata_map or {}
        try:
            class_to_idx = normalize_class_to_idx(json.loads(metadata.get("class_to_idx", "{}")))
        except json.JSONDecodeError:
            class_to_idx = DEFAULT_CLASS_TO_IDX.copy()
        try:
            image_size = int(metadata.get("image_size", IMAGE_SIZE))
        except (TypeError, ValueError):
            image_size = IMAGE_SIZE
        input_name = session.get_inputs()[0].name
        loaded = (session, class_to_idx, image_size, input_name)
        self.onnx_cache[cache_key] = loaded
        return loaded

    def predict(self, weight_path: str | Path, image: Image.Image) -> dict[str, float]:
        """Run inference with the selected .pth or .onnx file."""
        path = Path(weight_path)
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy weight: {path}")

        suffix = path.suffix.lower()
        if suffix == ".pth":
            torch = self._torch()
            model, class_to_idx, image_size, device = self._load_pth(path)
            input_array = preprocess_image(image, image_size)
            input_tensor = torch.from_numpy(input_array).to(device)
            with torch.inference_mode():
                logits = model(input_tensor).detach().cpu().numpy()
        elif suffix == ".onnx":
            session, class_to_idx, image_size, input_name = self._load_onnx(path)
            input_array = preprocess_image(image, image_size)
            logits = session.run(None, {input_name: input_array})[0]
        else:
            raise ValueError("Weight phải có đuôi .pth hoặc .onnx.")

        probabilities = softmax(logits)[0]
        index_to_name = names_by_index(class_to_idx)
        return {
            index_to_name.get(index, f"class_{index}"): float(probability)
            for index, probability in enumerate(probabilities)
        }


def find_local_weights(weights_dir: Path) -> list[tuple[str, str]]:
    """Return Gradio dropdown choices for local model files."""
    if not weights_dir.is_dir():
        return []
    files = sorted(
        [*weights_dir.glob("*.pth"), *weights_dir.glob("*.onnx")],
        key=lambda path: path.name,
    )
    return [(path.name, str(path)) for path in files]


def build_interface(engine: InferenceEngine, weights_dir: Path) -> Any:
    """Build the Gradio interface."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Chưa cài Gradio. Hãy chạy: pip install gradio"
        ) from exc

    choices = find_local_weights(weights_dir)
    default_weight = choices[0][1] if choices else None

    def classify(
        image: Image.Image | None,
        selected_weight: str | None,
        uploaded_weight: str | None,
    ) -> tuple[dict[str, float] | None, str]:
        if image is None:
            return None, "Hãy upload một ảnh trước."
        weight_path = uploaded_weight or selected_weight
        if not weight_path:
            return None, "Hãy chọn hoặc upload một weight `.pth`/`.onnx`."
        try:
            scores = engine.predict(weight_path, image)
            predicted = max(scores, key=scores.get)
            confidence = scores[predicted] * 100.0
            status = f"Kết quả: **{predicted}** — độ tin cậy **{confidence:.2f}%**"
            return scores, status
        except Exception as exc:  # Show a useful message inside the UI.
            return None, f"Lỗi: `{type(exc).__name__}: {exc}`"

    with gr.Blocks(title="Dodo classifier demo") as demo:
        gr.Markdown(
            "# Dodo / no_dodo classifier\n"
            "Chọn weight `.pth` hoặc `.onnx`, upload ảnh, rồi bấm **Phân loại**."
        )
        with gr.Row():
            with gr.Column():
                selected_weight = gr.Dropdown(
                    choices=choices,
                    value=default_weight,
                    label="Weight có sẵn",
                    info="Các file trong thư mục checkpoints.",
                )
                uploaded_weight = gr.File(
                    label="Hoặc upload weight mới (.pth/.onnx)",
                    file_types=[".pth", ".onnx"],
                    type="filepath",
                )
                image = gr.Image(type="pil", label="Upload ảnh cần phân loại")
                classify_button = gr.Button("Phân loại", variant="primary")
            with gr.Column():
                result = gr.Label(label="Xác suất", num_top_classes=2)
                status = gr.Markdown("Chưa có kết quả.")

        classify_button.click(
            fn=classify,
            inputs=[image, selected_weight, uploaded_weight],
            outputs=[result, status],
        )
    return demo


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("--port phải nằm trong khoảng 1 đến 65535.")
    engine = InferenceEngine(device_request=args.device)
    demo = build_interface(engine, args.weights_dir)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
