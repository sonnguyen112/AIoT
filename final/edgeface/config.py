from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    mqtt_host: str = field(default_factory=lambda: os.getenv("MQTT_HOST", "127.0.0.1"))
    mqtt_port: int = field(default_factory=lambda: int(os.getenv("MQTT_PORT", "1883")))
    mqtt_username: str = field(default_factory=lambda: os.getenv("MQTT_USERNAME", "edgeface_pi"))
    mqtt_password: str = field(default_factory=lambda: os.getenv("MQTT_PASSWORD", ""), repr=False)
    device_id: str = field(default_factory=lambda: os.getenv("DEVICE_ID", "raspberrypi-01"))
    prefix: str = field(default_factory=lambda: os.getenv("TOPIC_PREFIX", "aiot/edgeface"))
    dataset_dir: Path = field(default_factory=lambda: Path(os.getenv("DATASET_DIR", "dataset")))
    model_dir: Path = field(default_factory=lambda: Path(os.getenv("MODEL_DIR", "models/active")))
    display_names_path: Path = field(default_factory=lambda: Path(os.getenv("DISPLAY_NAMES_PATH", "configs/display_names.json")))
    detector_path: Path = field(default_factory=lambda: Path(os.getenv("DETECTOR_PATH", "models/face_detection_yunet_2023mar.onnx")))
    camera_backend: str = field(default_factory=lambda: os.getenv("CAMERA_BACKEND", "auto"))
    camera_index: int = field(default_factory=lambda: int(os.getenv("CAMERA_INDEX", "0")))
    camera_width: int = field(default_factory=lambda: int(os.getenv("CAMERA_WIDTH", "640")))
    camera_height: int = field(default_factory=lambda: int(os.getenv("CAMERA_HEIGHT", "480")))
    frame_interval: float = field(default_factory=lambda: float(os.getenv("FRAME_INTERVAL_S", "0.15")))
    ort_threads: int = field(default_factory=lambda: int(os.getenv("ORT_THREADS", "2")))
    rejection_threshold: float | None = field(default_factory=lambda: (
        float(os.environ["REJECTION_THRESHOLD"]) if os.getenv("REJECTION_THRESHOLD", "").strip() else None))
    collection_target: int = field(default_factory=lambda: int(os.getenv("COLLECTION_TARGET", "150")))
    collection_interval: float = field(default_factory=lambda: float(os.getenv("COLLECTION_INTERVAL_S", "0.5")))
    collection_max_seconds: int = field(default_factory=lambda: int(os.getenv("COLLECTION_MAX_SECONDS", "600")))
    min_face_size: int = field(default_factory=lambda: int(os.getenv("MIN_FACE_SIZE", "80")))
    min_blur: float = field(default_factory=lambda: float(os.getenv("MIN_BLUR_VARIANCE", "60")))
    min_brightness: float = field(default_factory=lambda: float(os.getenv("MIN_BRIGHTNESS", "35")))
    max_brightness: float = field(default_factory=lambda: float(os.getenv("MAX_BRIGHTNESS", "220")))
    duplicate_hamming: int = field(default_factory=lambda: int(os.getenv("DUPLICATE_HAMMING", "4")))
    cooldown: float = field(default_factory=lambda: float(os.getenv("NOTIFICATION_COOLDOWN_S", "30")))
    telemetry_interval: float = field(default_factory=lambda: float(os.getenv("TELEMETRY_INTERVAL_S", "10")))
    no_face_after: float = field(default_factory=lambda: float(os.getenv("NO_FACE_AFTER_S", "8")))
    auto_stop_known: bool = field(default_factory=lambda: os.getenv("AUTO_STOP_ON_KNOWN", "true").lower() == "true")
    command_timeout: float = field(default_factory=lambda: float(os.getenv("COMMAND_TIMEOUT_S", "15")))
    command_ttl: float = field(default_factory=lambda: float(os.getenv("COMMAND_TTL_S", "60")))
    audio_enabled: bool = field(default_factory=lambda: os.getenv("AUDIO_ENABLED", "false").lower() == "true")
    audio_language: str = field(default_factory=lambda: os.getenv("AUDIO_LANGUAGE", "vi"))
    recognition_photo_enabled: bool = field(default_factory=lambda: os.getenv("RECOGNITION_PHOTO_ENABLED", "false").lower() == "true")
    snapshot_token: str = field(default_factory=lambda: os.getenv("SNAPSHOT_TOKEN", ""), repr=False)
    snapshot_base_url: str = field(default_factory=lambda: os.getenv("SNAPSHOT_BASE_URL", ""))
    snapshot_bind: str = field(default_factory=lambda: os.getenv("SNAPSHOT_BIND", "127.0.0.1"))
    snapshot_port: int = field(default_factory=lambda: int(os.getenv("SNAPSHOT_PORT", "8765")))
    snapshot_ttl: float = field(default_factory=lambda: float(os.getenv("SNAPSHOT_TTL_S", "120")))
    snapshot_timeout: float = field(default_factory=lambda: float(os.getenv("SNAPSHOT_TIMEOUT_S", "5")))
    snapshot_font: Path = field(default_factory=lambda: Path(os.getenv("SNAPSHOT_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")))

    @classmethod
    def load(cls) -> Config:
        load_dotenv()
        cfg = cls()
        if not cfg.mqtt_password:
            raise ValueError("MQTT_PASSWORD is required (anonymous MQTT is disabled)")
        if cfg.ort_threads < 1 or cfg.frame_interval <= 0 or cfg.collection_interval <= 0:
            raise ValueError("Threads and sampling intervals must be positive")
        if cfg.rejection_threshold is not None and not 0 <= cfg.rejection_threshold <= 1:
            raise ValueError("REJECTION_THRESHOLD must be between 0 and 1, or empty for validation calibration")
        if cfg.collection_target < 1 or cfg.collection_target > 1000:
            raise ValueError("COLLECTION_TARGET must be between 1 and 1000")
        if cfg.camera_backend not in {"auto", "usb", "csi"}:
            raise ValueError("CAMERA_BACKEND must be auto, usb or csi")
        if cfg.recognition_photo_enabled and len(cfg.snapshot_token) < 32:
            raise ValueError("Photo notifications require SNAPSHOT_TOKEN with at least 32 characters")
        if cfg.snapshot_ttl <= 0 or cfg.snapshot_timeout <= 0 or not 1 <= cfg.snapshot_port <= 65535:
            raise ValueError("Invalid snapshot TTL, timeout or port")
        return cfg
