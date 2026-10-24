from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, Update
from telegram.error import TelegramError
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from edgeface.config import Config
from edgeface.mqtt import MQTTLink
from edgeface.names import load_display_names
from edgeface.preprocess import save_json
from edgeface.protocol import decode, make_command, safe_name
from edgeface.snapshot import fetch_snapshot

log = logging.getLogger(__name__)
START_BUTTON = "📷 Bắt đầu nhận diện"
CONTROL_KEYBOARD = ReplyKeyboardMarkup(
    [[START_BUTTON]], resize_keyboard=True, is_persistent=True,
    input_field_placeholder="Bấm nút để bắt đầu nhận diện",
)
HELP = ("Bấm nút 📷 Bắt đầu nhận diện để mở camera và nhận diện trên Pi.\n\n"
        "EdgeFace commands:\n/start — enable local Pi recognition\n/stop — release camera\n"
        "/status — device health\n/members — registered members\n"
        "/collect <name> — collect after consent confirmation\n"
        "/collect_unknown <anonymous-id> — one consenting unknown participant per session\n"
        "/help — commands\nWhen enabled, recognized-member photos are sent to the chat that started recognition. "
        "This prototype has no liveness detection.")


def ids(key: str) -> set[int]:
    return {int(item.strip()) for item in os.getenv(key, "").split(",") if item.strip()}


class Bot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.display_names = load_display_names(cfg.display_names_path)
        self.users = ids("TELEGRAM_ALLOWED_USER_IDS")
        self.chats = ids("TELEGRAM_ALLOWED_CHAT_IDS")
        self.subscribers: set[int] = set(self.chats)
        try:
            previous = json.loads(Path("reports/generated/bot_subscribers.json").read_text())
            self.subscribers.update(int(chat) for chat in previous if not self.chats or int(chat) in self.chats)
        except (OSError, ValueError, TypeError):
            pass
        self.link = MQTTLink(cfg, "bot", self.receive)
        self.loop: asyncio.AbstractEventLoop | None = None
        self.queue: asyncio.Queue[tuple[str, dict[str, Any], bool]] = asyncio.Queue(maxsize=128)
        self.pending: dict[str, asyncio.Future] = {}
        self.consent: dict[str, tuple[int, int, float, dict[str, Any]]] = {}
        self.seen: OrderedDict[str, None] = OrderedDict()
        self.consumer: asyncio.Task | None = None
        self.recognition_chats: OrderedDict[str, int] = OrderedDict()

    def allowed(self, update: Update) -> bool:
        return bool(update.effective_user and update.effective_chat and
                    (not self.users or update.effective_user.id in self.users) and
                    (not self.chats or update.effective_chat.id in self.chats))

    def receive(self, topic: str, payload: bytes, retained: bool) -> None:
        value = decode(payload)
        if value.get("device_id") != self.cfg.device_id or not self.loop:
            return
        def enqueue() -> None:
            # Resolve acknowledgements directly on the asyncio loop: the consumer
            # can await an auto-stop while MQTT callbacks still deliver its ack.
            if topic == "status" and value.get("event") == "ack" and not retained:
                future = self.pending.get(value.get("request_id"))
                if future and not future.done():
                    future.set_result(value)
                return
            try:
                self.queue.put_nowait((topic, value, retained))
            except asyncio.QueueFull:
                log.warning("Bot event queue full")
        self.loop.call_soon_threadsafe(enqueue)

    async def init(self, app: Application) -> None:
        self.loop = asyncio.get_running_loop()
        self.link.start()
        self.consumer = asyncio.create_task(self.consume(app))

    async def close(self, app: Application) -> None:
        if self.consumer:
            self.consumer.cancel()
            await asyncio.gather(self.consumer, return_exceptions=True)
        self.link.close()

    async def consume(self, app: Application) -> None:
        while True:
            try:
                topic, value, retained = await asyncio.wait_for(self.queue.get(), timeout=5)
            except asyncio.TimeoutError:
                save_json(Path("reports/generated/bot_health.json"),
                          {"timestamp": time.time(), "mqtt_connected": self.link.connected.is_set()})
                continue
            if topic == "status" and value.get("event") == "ack" and not retained:
                future = self.pending.get(value.get("request_id"))
                if future and not future.done():
                    future.set_result(value)
                continue
            if retained or topic not in {"result", "collection"}:
                continue
            identity = json.dumps(value, sort_keys=True)
            if identity in self.seen:
                continue
            self.seen[identity] = None
            if len(self.seen) > 512:
                self.seen.popitem(last=False)
            await self.notify_event(app, topic, value)

    async def notify_event(self, app: Application, topic: str, value: dict[str, Any]) -> None:
        text = self.event_text(topic, value)
        if not text:
            return
        if topic == "result":
            chat = self.recognition_chats.get(value.get("request_id"))
            recipients = (chat,) if chat is not None else ()
        else:
            recipients = tuple(self.subscribers)
        # The snapshot is already in Pi RAM; stop the camera before any network upload.
        text += await self.auto_stop_result(topic, value)
        for chat in recipients:
            try:
                await app.bot.send_message(chat, text)
                log.info("Telegram result text sent (request_id=%s)", value.get("request_id"))
            except TelegramError:
                log.warning("Telegram text notification failed")
        if (not recipients or not self.cfg.recognition_photo_enabled or topic != "result"
                or value.get("event") != "recognition" or value.get("person") == "Unknown"):
            return
        try:
            image_id = value["image"]["id"]
            if not image_id:
                raise ValueError("Snapshot is unavailable")
            photo = await asyncio.to_thread(fetch_snapshot, self.cfg, image_id)
        except Exception as exc:
            log.warning("Recognition photo download failed (%s)", type(exc).__name__)
            for chat in recipients:
                try:
                    await app.bot.send_message(chat, "Không tải được ảnh nhận diện. Kết quả đã gửi bằng tin nhắn ở trên.")
                except TelegramError:
                    log.warning("Telegram photo fallback notification failed")
            return
        caption = f"{self.display_name(value['person'])} · Confidence: {value['confidence']:.1%}"
        for chat in recipients:
            try:
                # Separate photo and text messages, as requested.
                await app.bot.send_photo(chat_id=chat, photo=photo, caption=caption,
                                         filename="edgeface-recognition.jpg")
                log.info("Telegram recognition photo sent (request_id=%s)", value.get("request_id"))
            except TelegramError:
                log.warning("Telegram recognition photo upload failed")
                try:
                    await app.bot.send_message(chat, "Không gửi được ảnh; kết quả nhận diện đã được gửi bằng tin nhắn.")
                except TelegramError:
                    log.warning("Telegram photo fallback notification failed")

    async def auto_stop_result(self, topic: str, value: dict[str, Any]) -> str:
        person = value.get("person")
        if (not self.cfg.auto_stop_known or topic != "result" or value.get("event") != "recognition"
                or not isinstance(person, str) or not person or person == "Unknown"):
            return ""
        request = value.get("request_id")
        if not request:
            return "\nAuto-stop skipped: recognition session ID is missing."
        try:
            response = await self.dispatch(make_command("stop", self.cfg.device_id, expected_request_id=request))
            if response.get("stop_skipped"):
                return "\nRecognition came from an earlier session; current session continues."
            if response.get("ok") and response.get("state") == "idle":
                return "\nCamera stopped automatically. Use /start to recognize again."
            return "\nAuto-stop was not confirmed. Check /status."
        except RuntimeError:
            log.warning("Automatic stop after known recognition was not confirmed")
            return "\nAuto-stop was not confirmed. Check /status."

    def display_name(self, person: str) -> str:
        return self.display_names.get(person, person)

    def event_text(self, topic: str, value: dict[str, Any]) -> str | None:
        event = value.get("event")
        if topic == "collection":
            if event == "progress":
                # Pi sends progress every 3 seconds; Telegram only each 25 images.
                captured = value.get("captured", 0)
                if not captured or captured % 25:
                    return None
                return f"Collection: {self.display_name(value['person'])} — {captured}/{value['target']}"
            if event in {"completed", "aborted", "timeout", "error"}:
                return (f"Collection {event}.\nPerson: {self.display_name(value['person'])}\nCaptured: {value['captured']} images\n"
                        f"Rejected: {value['rejected']} frames\nStorage: {value['storage']}")
            return None
        if event == "no_face":
            return "No face detected. Camera remains active."
        if event == "multiple_faces":
            return f"{value['face_count']} faces detected. Use one person at a time."
        if event == "recognition":
            if value["person"] == "Unknown":
                return None
            title = "Recognition successful!"
            return (f"{title}\nPerson: {self.display_name(value['person'])}\nConfidence: {value['confidence']:.1%}\n"
                    f"Device: {value['device_id']}\nInference: {value['inference_ms']:.1f} ms")
        return None

    async def dispatch(self, command: dict[str, Any]) -> dict[str, Any]:
        if not self.link.connected.is_set():
            raise RuntimeError("MQTT unavailable; command not sent")
        request = command["request_id"]
        future = asyncio.get_running_loop().create_future()
        self.pending[request] = future
        try:
            if not self.link.publish("command", command):
                raise RuntimeError("MQTT command publish failed")
            return await asyncio.wait_for(future, timeout=self.cfg.command_timeout)
        except asyncio.TimeoutError as exc:
            raise RuntimeError("Pi did not respond before timeout. Use /status; command may have arrived.") from exc
        finally:
            self.pending.pop(request, None)

    async def command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self.allowed(update):
            return
        message = update.effective_message
        if not message:
            return
        action = "start" if message.text == START_BUTTON else message.text.split()[0].split("@")[0][1:]
        if action in {"start", "collect", "collect_unknown"}:
            self.subscribers.add(update.effective_chat.id)
            save_json(Path("reports/generated/bot_subscribers.json"), sorted(self.subscribers))
        if action == "help":
            await message.reply_text(HELP, reply_markup=CONTROL_KEYBOARD)
            return
        args: dict[str, Any] = {}
        if action in {"collect", "collect_unknown"}:
            try:
                name = safe_name(" ".join(context.args))
                if action == "collect" and name.lower() == "unknown":
                    raise ValueError("Use /collect_unknown <anonymous-id>")
                args = {"name": name} if action == "collect" else {"identity": name}
            except ValueError as exc:
                await message.reply_text(f"{exc}\nUsage: /{action} <{'name' if action == 'collect' else 'anonymous-id'}>")
                return
            command = make_command(action, self.cfg.device_id, target=self.cfg.collection_target, **args)
            key = command["request_id"]
            self.consent = {k: v for k, v in self.consent.items() if time.monotonic()-v[2] < 120}
            self.consent[key] = (update.effective_user.id, update.effective_chat.id, time.monotonic(), command)
            keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Participant agreed — start", callback_data=f"consent:{key}")]])
            await message.reply_text("Confirm that this participant agreed to local face collection and model training.\n"
                                     "Keep other people out of the camera view.", reply_markup=keyboard)
            return
        await self.execute(message, make_command(action, self.cfg.device_id))

    async def consent_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self.allowed(update) or not update.callback_query:
            return
        query = update.callback_query
        key = query.data.split(":", 1)[1]
        pending = self.consent.get(key)
        if not pending or pending[:2] != (update.effective_user.id, update.effective_chat.id) or time.monotonic()-pending[2] > 120:
            await query.answer("Confirmation expired. Repeat collection command.")
            return
        self.consent.pop(key)
        await query.answer()
        command = make_command(pending[3]["action"], self.cfg.device_id, consent=True,
                               **{k: v for k, v in pending[3].items() if k in {"name", "identity", "target"}})
        await self.execute(query.message, command)

    async def execute(self, message: Any, command: dict[str, Any]) -> None:
        # Register before publishing: a fast Pi result may arrive before its start ACK.
        if command["action"] == "start" and isinstance(getattr(message, "chat_id", None), int):
            self.recognition_chats[command["request_id"]] = message.chat_id
            while len(self.recognition_chats) > 128:
                self.recognition_chats.popitem(last=False)
        try:
            response = await self.dispatch(command)
            if not response.get("ok"):
                self.recognition_chats.pop(command["request_id"], None)
                reason = response.get("command_error") or (response.get("error") or {}).get("message", "Unknown device error")
                await message.reply_text(f"Pi refused command: {reason}", reply_markup=CONTROL_KEYBOARD)
                return
            action = command["action"]
            if action == "start":
                text = "Đang nhận diện..."
            elif action == "stop":
                text = "Đã dừng nhận diện."
            elif action in {"collect", "collect_unknown"}:
                text = f"Started collecting face images for {self.display_name(response['person'])}.\nTarget: {response['target']} valid images."
            elif action == "members":
                text = "Model members: " + (", ".join(self.display_name(n) for n in response["members"]) or "no trained model")
                text += "\nCollected: " + (", ".join(self.display_name(n) for n in response["collected_members"]) or "none")
            elif action == "status":
                def metric(key: str, suffix: str) -> str:
                    v = response.get(key)
                    return "n/a" if v is None else f"{v:.1f}{suffix}"
                text = (f"Device: {response['device_id']}\nState: {response['state']}\n"
                        f"Camera: {response.get('camera') or 'closed'}\nModel: {response.get('model_version') or 'not deployed'}\n"
                        f"CPU: {metric('cpu_percent', '%')}\nTemperature: {metric('cpu_temperature_c', ' °C')}\n"
                        f"RAM: {metric('ram_mb', ' MB')}\nInference: {metric('inference_ms', ' ms')}\n"
                        f"Error: {response.get('error') or 'none'}")
            else:
                text = f"Device {response['device_id']}: {response['state']}"
            await message.reply_text(text, reply_markup=CONTROL_KEYBOARD)
        except RuntimeError as exc:
            await message.reply_text(str(exc), reply_markup=CONTROL_KEYBOARD)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.error("Telegram handler failed (%s)", type(context.error).__name__)


def main() -> None:
    os.umask(0o077)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    cfg = Config.load()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN is required")
    bot = Bot(cfg)
    app = Application.builder().token(token).post_init(bot.init).post_stop(bot.close).build()
    for action in ("start", "stop", "collect", "collect_unknown", "status", "members", "help"):
        app.add_handler(CommandHandler(action, bot.command))
    app.add_handler(MessageHandler(filters.Text([START_BUTTON]), bot.command))
    app.add_handler(CallbackQueryHandler(bot.consent_callback, pattern="^consent:"))
    app.add_error_handler(error_handler)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
