from __future__ import annotations

import json
import logging
import os
import queue
import signal
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from edgeface.config import Config
from edgeface.mqtt import MQTTLink
from edgeface.names import load_display_names
from edgeface.preprocess import save_json
from edgeface.protocol import Cooldown, decode, now_iso, validate_command
from raspberry_pi.audio import Speaker
from raspberry_pi.camera import Camera, CameraError
from raspberry_pi.collection import Collector
from raspberry_pi.detection import DetectionError, FaceDetector, crop_face
from raspberry_pi.inference import Classifier, ModelError
from raspberry_pi.snapshots import SnapshotService

log = logging.getLogger(__name__)


def temperature() -> float | None:
    try:
        return float(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except (OSError, ValueError):
        return None


class Agent:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.display_names = load_display_names(cfg.display_names_path)
        self.link = MQTTLink(cfg, "pi", self.on_message)
        self.commands: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=32)
        self.shutdown = threading.Event()
        self.camera = Camera(cfg)
        self.detector: FaceDetector | None = None
        self.classifier: Classifier | None = None
        self.collector: Collector | None = None
        self.speaker = Speaker(cfg.audio_enabled, cfg.audio_language)
        self.snapshots = SnapshotService(cfg)
        self.cooldown = Cooldown(cfg.cooldown)
        self.state = "idle"
        self.error: dict[str, str] | None = None
        self.active_request: str | None = None
        self.last_face = time.monotonic()
        self.last_progress = 0.0
        self.last_telemetry = 0.0
        self.inference_ms: float | None = None
        self.detection_ms: float | None = None
        self.e2e_ms: float | None = None
        cfg.dataset_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        cfg.dataset_dir.chmod(0o700)
        db_path = cfg.dataset_dir / "requests.sqlite3"
        self.db = sqlite3.connect(db_path)
        db_path.chmod(0o600)
        self.db.execute("CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, response TEXT, created REAL)")
        self.db.commit()
        self.process = psutil.Process()
        psutil.cpu_percent(interval=None)

    def envelope(self, **kwargs: Any) -> dict[str, Any]:
        return {"device_id": self.cfg.device_id, "timestamp": now_iso(), **kwargs}

    def status(self, **kwargs: Any) -> dict[str, Any]:
        return self.envelope(online=True, state=self.state, camera=self.camera.backend,
                             model_version=self.classifier.version if self.classifier else None,
                             model_bundle_version=self.classifier.bundle_version if self.classifier else None,
                             rejection_threshold=self.classifier.threshold if self.classifier else self.cfg.rejection_threshold,
                             rejection_threshold_source=("manual_override" if self.cfg.rejection_threshold is not None else "validation"),
                             inference_ms=self.inference_ms, detection_ms=self.detection_ms,
                             end_to_end_ms=self.e2e_ms, cpu_percent=psutil.cpu_percent(),
                             cpu_temperature_c=temperature(), ram_mb=self.process.memory_info().rss / 1024**2,
                             error=self.error, **kwargs)

    def on_message(self, topic: str, payload: bytes, retained: bool) -> None:
        if topic != "command":
            return
        try:
            if retained:
                raise ValueError("Retained commands are forbidden")
            command = validate_command(decode(payload), self.cfg.command_ttl, self.cfg.device_id)
            self.commands.put_nowait(command)
        except (ValueError, TypeError, queue.Full) as exc:
            log.warning("Command refused: %s", exc)
            try:
                request_id = decode(payload).get("request_id")
            except Exception:
                request_id = None
            self.link.publish("status", self.envelope(event="ack", request_id=request_id, ok=False,
                                                     state=self.state, error={"kind": "command", "message": str(exc)}))

    def _stop(self, outcome: str = "aborted") -> None:
        try:
            self.camera.close()
        finally:
            if self.collector:
                self.collector.finish(outcome)
                self.link.publish("collection", self.envelope(**self.collector.progress(outcome)))
                self.collector = None
            self.state = "idle"
            self.active_request = None

    def handle(self, command: dict[str, Any]) -> None:
        request_id = command["request_id"]
        row = self.db.execute("SELECT response FROM requests WHERE id=?", (request_id,)).fetchone()
        if row:
            response = json.loads(row[0]) if row[0] else self.status(event="ack", request_id=request_id, ok=False,
                                                    error_message="Already processed before restart; query status")
            response["duplicate"] = True
            self.link.publish("status", response)
            return
        # Durable claim committed before opening camera or mutating collection data.
        self.db.execute("INSERT INTO requests VALUES (?, NULL, ?)", (request_id, time.time()))
        self.db.commit()
        action = command["action"]
        ok = True
        extras: dict[str, Any] = {}
        try:
            validate_command(command, self.cfg.command_ttl, self.cfg.device_id)
            if action == "stop":
                expected_request = command.get("expected_request_id")
                if expected_request and expected_request != self.active_request:
                    extras["stop_skipped"] = "Recognition belongs to an older session"
                else:
                    self._stop()
                    self.error = None
            elif action in {"start", "collect", "collect_unknown"}:
                if self.state in {"collecting", "recognizing"}:
                    raise ValueError("Device busy; use /stop before switching operation")
                self.error = None
                if self.detector is None:
                    self.detector = FaceDetector(str(self.cfg.detector_path))
                if action == "start":
                    self.classifier = Classifier(self.cfg.model_dir, self.cfg.ort_threads,
                                                 threshold_override=self.cfg.rejection_threshold)
                self.camera.open()
                self.active_request = request_id
                self.last_face = time.monotonic()
                if action == "start":
                    self.state = "recognizing"
                    # An explicit new session should report promptly even if the
                    # same member was recognized in a recently stopped session.
                    self.cooldown = Cooldown(self.cfg.cooldown)
                else:
                    self.collector = Collector(self.cfg, command)
                    self.collector.meta["camera"] = self.camera.backend
                    self.collector.meta["capture_pixel_format"] = getattr(self.camera, "pixel_format", None)
                    self.collector.meta["stored_color_order"] = "BGR_to_JPEG"
                    self.state = "collecting"
                    extras = self.collector.progress("started")
                    extras.pop("event")
                    extras.pop("timestamp")
                    extras.pop("request_id")
            elif action == "members":
                model_members: list[str] = []
                if (self.cfg.model_dir / "classes.json").exists():
                    model_members = [name for name in json.loads((self.cfg.model_dir / "classes.json").read_text()) if name != "Unknown"]
                raw = self.cfg.dataset_dir / "raw"
                extras = {"members": model_members, "collected_members": sorted(p.name for p in raw.iterdir() if p.is_dir() and p.name != "Unknown") if raw.exists() else []}
        except ValueError as exc:
            ok = False
            extras["command_error"] = str(exc)
        except Exception as exc:
            ok = False
            self.fail(exc)
        response = self.status(event="ack", request_id=request_id, action=action, ok=ok, **extras)
        self.db.execute("UPDATE requests SET response=? WHERE id=?", (json.dumps(response), request_id))
        self.db.commit()
        self.link.publish("status", response, retain=False)
        self.link.publish("status", self.status(), retain=True)

    def fail(self, exc: Exception) -> None:
        kind = "camera" if isinstance(exc, CameraError) else "model" if isinstance(exc, ModelError) else "detector" if isinstance(exc, DetectionError) else "runtime"
        log.exception("Device %s error", kind)
        try:
            self._stop("error")
        except Exception:
            log.exception("Cleanup failed")
        self.state = "error"
        self.error = {"kind": kind, "message": str(exc)}
        self.link.publish("status", self.status(), retain=True)

    def frame(self) -> None:
        start = time.perf_counter()
        frame = self.camera.read()
        detection_start = time.perf_counter()
        assert self.detector is not None
        faces = self.detector.detect(frame)
        self.detection_ms = (time.perf_counter() - detection_start) * 1000
        now = time.monotonic()
        if self.collector and now - self.collector.started >= self.cfg.collection_max_seconds:
            self._stop("timeout")
            return
        if len(faces) != 1:
            event = "no_face" if not faces else "multiple_faces"
            if self.collector:
                self.collector.rejections[event] += 1
            else:
                if faces:
                    self.last_face = now
                if (faces or now-self.last_face >= self.cfg.no_face_after) and self.cooldown.allow(event, now):
                    self.link.publish("result", self.envelope(request_id=self.active_request, event=event, face_count=len(faces)))
        else:
            self.last_face = now
            crop = crop_face(frame, faces[0])
            if self.collector:
                self.collector.offer(crop, float(min(faces[0][2:4])), now)
                if self.collector.accepted >= self.collector.target:
                    self._stop("completed")
            else:
                assert self.classifier is not None
                person, confidence, self.inference_ms = self.classifier.predict(crop)
                self.e2e_ms = (time.perf_counter() - start) * 1000
                if self.cooldown.allow(person, now):
                    image = None
                    if person != "Unknown" and self.cfg.recognition_photo_enabled and self.link.connected.is_set():
                        try:
                            image = self.snapshots.add(frame, faces[0], self.display_names.get(person, person), confidence)
                        except Exception as exc:
                            # Image delivery is optional: preserve recognition and auto-stop on failure.
                            log.warning("Recognition snapshot unavailable (%s)", type(exc).__name__)
                    self.link.publish("result", self.envelope(request_id=self.active_request, event="recognition",
                        person=person, confidence=confidence, inference_ms=self.inference_ms,
                        display_name=self.display_names.get(person, person),
                        detection_ms=self.detection_ms, end_to_end_ms=self.e2e_ms,
                        model_version=self.classifier.version, **({"image": image} if image else {})))
                    if person != "Unknown":
                        self.speaker.say(person)
        if self.collector and now-self.last_progress >= 3:
            self.last_progress = now
            self.link.publish("collection", self.envelope(**self.collector.progress()))

    def run(self) -> None:
        self.link.start()
        connected_before = False
        try:
            while not self.shutdown.is_set():
                now = time.monotonic()
                online = self.link.connected.is_set()
                if online and not connected_before:
                    if self.cfg.recognition_photo_enabled:
                        try:
                            self.snapshots.start()
                        except Exception as exc:
                            log.warning("Snapshot HTTP unavailable (%s)", type(exc).__name__)
                    self.link.publish("status", self.status(), retain=True)
                connected_before = online
                for _ in range(32):
                    try:
                        self.handle(self.commands.get_nowait())
                    except queue.Empty:
                        break
                if now-self.last_telemetry >= self.cfg.telemetry_interval:
                    self.last_telemetry = now
                    telemetry = self.status()
                    telemetry["mqtt_connected"] = online
                    self.link.publish("telemetry", telemetry)
                    self.link.publish("status", self.status(), retain=True)
                    save_json(Path("reports/generated/device_health.json"), telemetry)
                if self.state in {"recognizing", "collecting"}:
                    try:
                        self.frame()
                    except Exception as exc:
                        self.fail(exc)
                self.shutdown.wait(self.cfg.frame_interval)
        finally:
            self._stop()
            self.link.publish("status", self.envelope(online=False, state="offline"), retain=True)
            # Allow QoS 1 offline message to flush before disconnect.
            time.sleep(0.2)
            self.link.close()
            self.speaker.close()
            self.snapshots.close()
            self.db.close()


def main() -> None:
    os.umask(0o077)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    agent = Agent(Config.load())
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: agent.shutdown.set())
    agent.run()


if __name__ == "__main__":
    main()
