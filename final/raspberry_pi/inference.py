from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from edgeface.preprocess import PREPROCESS, bundle_version, reject, softmax, tensor_from_rgb


class ModelError(RuntimeError):
    pass


class Classifier:
    def __init__(self, directory: Path, threads: int = 2, *, threshold_override: float | None = None):
        try:
            self.classes = json.loads((directory / "classes.json").read_text())
            self.preprocessing = json.loads((directory / "preprocessing.json").read_text())
            calibration = json.loads((directory / "threshold.json").read_text())
            self.threshold = float(calibration["threshold"])
            if not 0 <= self.threshold <= 1 or calibration.get("source") != "validation":
                raise ValueError("Threshold must be calibrated on validation data")
            self.calibrated_threshold = self.threshold
            self.threshold_source = "validation"
            if threshold_override is not None:
                self.threshold = float(threshold_override)
                if not 0 <= self.threshold <= 1:
                    raise ValueError("Threshold override must be between 0 and 1")
                self.threshold_source = "manual_override"
            if not isinstance(self.classes, list) or "Unknown" not in self.classes or len(set(self.classes)) != len(self.classes):
                raise ValueError("Classes must be unique and include Unknown")
            if self.preprocessing != PREPROCESS:
                raise ValueError("Preprocessing mismatch")
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self.session = ort.InferenceSession(str(directory / "face_classifier.onnx"), options,
                                                providers=["CPUExecutionProvider"])
            inputs = self.session.get_inputs()
            if len(inputs) != 1 or inputs[0].shape != [1, 3, 224, 224] or inputs[0].type != "tensor(float)":
                raise ValueError("Expected fixed float32 input 1x3x224x224")
            output = self.session.get_outputs()[0]
            if output.shape != [1, len(self.classes)]:
                raise ValueError("Model classes/output mismatch")
            self.input_name = inputs[0].name
            self.version = hashlib.sha256((directory / "face_classifier.onnx").read_bytes()).hexdigest()[:12]
            self.bundle_version = bundle_version(directory)
        except Exception as exc:
            raise ModelError(str(exc)) from exc

    def predict(self, bgr: np.ndarray) -> tuple[str, float, float]:
        try:
            array = tensor_from_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.preprocessing)
            start = time.perf_counter()
            logits = self.session.run(None, {self.input_name: array})[0]
            latency = (time.perf_counter() - start) * 1000
            person, confidence = reject(softmax(logits)[0], self.classes, self.threshold)
            return person, confidence, latency
        except Exception as exc:
            raise ModelError(str(exc)) from exc
