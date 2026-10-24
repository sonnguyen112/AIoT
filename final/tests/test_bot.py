import asyncio
import json
from unittest.mock import AsyncMock
from dataclasses import replace
from types import SimpleNamespace

from edgeface.config import Config
from telegram_bot.bot import Bot, CONTROL_KEYBOARD, START_BUTTON


def bot(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TELEGRAM_ALLOWED_USER_IDS", raising=False)
    monkeypatch.delenv("TELEGRAM_ALLOWED_CHAT_IDS", raising=False)
    return Bot(Config(mqtt_password="test-only", auto_stop_known=True))


def test_known_recognition_stops_matching_session(monkeypatch, tmp_path):
    b = bot(monkeypatch, tmp_path)
    b.dispatch = AsyncMock(return_value={"ok": True, "state": "idle"})
    text = asyncio.run(b.auto_stop_result("result", {"event": "recognition", "person": "son", "request_id": "session-id"}))
    command = b.dispatch.call_args.args[0]
    assert command["action"] == "stop" and command["expected_request_id"] == "session-id"
    assert "Camera stopped automatically" in text


def test_unknown_and_no_face_do_not_stop(monkeypatch, tmp_path):
    b = bot(monkeypatch, tmp_path)
    b.dispatch = AsyncMock()
    async def check():
        assert await b.auto_stop_result("result", {"event": "recognition", "person": "Unknown", "request_id": "id"}) == ""
        assert await b.auto_stop_result("result", {"event": "no_face"}) == ""
    asyncio.run(check())
    b.dispatch.assert_not_called()


def test_auto_stop_waits_for_actual_ack_and_handles_timeout(monkeypatch, tmp_path):
    b = bot(monkeypatch, tmp_path)
    b.dispatch = AsyncMock(side_effect=RuntimeError("timeout"))
    text = asyncio.run(b.auto_stop_result("result", {"event": "recognition", "person": "son", "request_id": "session"}))
    assert "not confirmed" in text and "Camera stopped automatically" not in text


def test_ack_can_resolve_while_consumer_awaits_auto_stop(monkeypatch, tmp_path):
    b = bot(monkeypatch, tmp_path)
    async def check():
        b.loop = asyncio.get_running_loop()
        future = b.loop.create_future()
        b.pending["request"] = future
        b.receive("status", json.dumps({"device_id": b.cfg.device_id, "event": "ack", "request_id": "request", "ok": True}).encode(), False)
        response = await asyncio.wait_for(future, 0.2)
        assert response["ok"] and b.queue.empty()
    asyncio.run(check())


def test_result_displays_full_name_but_keeps_class_id(monkeypatch, tmp_path):
    b = bot(monkeypatch, tmp_path)
    b.display_names = {"son": "Nguyễn Hồ Trường Sơn"}
    value = {"event": "recognition", "person": "son", "confidence": 0.95,
             "device_id": "pi-test", "inference_ms": 30}
    text = b.event_text("result", value)
    assert "Person: Nguyễn Hồ Trường Sơn" in text
    assert value["person"] == "son"
    assert b.display_name("Unknown") == "Unknown"


def test_unknown_is_silent_in_telegram(monkeypatch, tmp_path):
    b = bot(monkeypatch, tmp_path)
    assert b.event_text("result", {"event": "recognition", "person": "Unknown"}) is None


def test_start_shows_recognizing_status(monkeypatch, tmp_path):
    b = bot(monkeypatch, tmp_path)
    b.dispatch = AsyncMock(return_value={"ok": True, "state": "recognizing"})
    message = type("Message", (), {"reply_text": AsyncMock()})()
    asyncio.run(b.execute(message, {"action": "start"}))
    message.reply_text.assert_awaited_once_with("Đang nhận diện...", reply_markup=CONTROL_KEYBOARD)


def test_start_button_sends_mqtt_start_and_routes_result_to_pressing_chat(monkeypatch, tmp_path):
    b = bot(monkeypatch, tmp_path)
    b.dispatch = AsyncMock(return_value={"ok": True, "state": "recognizing"})
    message = SimpleNamespace(text=START_BUTTON, chat_id=123, reply_text=AsyncMock())
    update = SimpleNamespace(effective_message=message,
        effective_user=SimpleNamespace(id=42), effective_chat=SimpleNamespace(id=123))
    asyncio.run(b.command(update, SimpleNamespace(args=[])))
    command = b.dispatch.call_args.args[0]
    assert command["action"] == "start" and command["device_id"] == b.cfg.device_id
    assert b.recognition_chats[command["request_id"]] == 123
    assert 123 in b.subscribers
    message.reply_text.assert_awaited_once_with("Đang nhận diện...", reply_markup=CONTROL_KEYBOARD)


def test_start_button_respects_telegram_allowlist(monkeypatch, tmp_path):
    b = bot(monkeypatch, tmp_path)
    b.users = {999}
    b.dispatch = AsyncMock()
    message = SimpleNamespace(text=START_BUTTON, chat_id=123, reply_text=AsyncMock())
    update = SimpleNamespace(effective_message=message,
        effective_user=SimpleNamespace(id=42), effective_chat=SimpleNamespace(id=123))
    asyncio.run(b.command(update, SimpleNamespace(args=[])))
    b.dispatch.assert_not_called()
    message.reply_text.assert_not_called()


def test_help_shows_button_without_starting_camera(monkeypatch, tmp_path):
    b = bot(monkeypatch, tmp_path)
    b.dispatch = AsyncMock()
    message = SimpleNamespace(text="/help", chat_id=123, reply_text=AsyncMock())
    update = SimpleNamespace(effective_message=message,
        effective_user=SimpleNamespace(id=42), effective_chat=SimpleNamespace(id=123))
    asyncio.run(b.command(update, SimpleNamespace(args=[])))
    b.dispatch.assert_not_called()
    assert message.reply_text.call_args.kwargs["reply_markup"] == CONTROL_KEYBOARD


def test_mqtt_failure_keeps_button_for_retry(monkeypatch, tmp_path):
    b = bot(monkeypatch, tmp_path)
    b.dispatch = AsyncMock(side_effect=RuntimeError("MQTT unavailable; command not sent"))
    message = SimpleNamespace(chat_id=123, reply_text=AsyncMock())
    asyncio.run(b.execute(message, {"action": "start", "request_id": "session"}))
    assert "MQTT unavailable" in message.reply_text.call_args.args[0]
    assert message.reply_text.call_args.kwargs["reply_markup"] == CONTROL_KEYBOARD


def test_known_sends_separate_text_and_photo_only_to_starting_chat(monkeypatch, tmp_path):
    b = bot(monkeypatch, tmp_path)
    b.cfg = replace(b.cfg, recognition_photo_enabled=True)
    b.recognition_chats["session"] = 123
    b.subscribers = {123, 999}
    b.dispatch = AsyncMock(return_value={"ok": True, "state": "idle"})
    calls = []
    async def text(chat, text):
        calls.append(("text", chat, text))
    async def photo(**kwargs):
        calls.append(("photo", kwargs["chat_id"], kwargs["photo"]))
    app = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock(side_effect=text), send_photo=AsyncMock(side_effect=photo)))
    monkeypatch.setattr("telegram_bot.bot.fetch_snapshot", lambda *args: b"jpeg-bytes")
    value = {"event": "recognition", "request_id": "session", "person": "son", "confidence": .94,
             "device_id": "test", "inference_ms": 30, "image": {"id": "a"*32}}
    asyncio.run(b.notify_event(app, "result", value))
    assert [c[:2] for c in calls] == [("text", 123), ("photo", 123)]
    assert "Confidence: 94.0%" in calls[0][2]
    assert b.dispatch.call_args.args[0]["expected_request_id"] == "session"


def test_photo_download_failure_preserves_text_and_auto_stop(monkeypatch, tmp_path):
    b = bot(monkeypatch, tmp_path)
    b.cfg = replace(b.cfg, recognition_photo_enabled=True)
    b.recognition_chats["session"] = 123
    b.dispatch = AsyncMock(return_value={"ok": True, "state": "idle"})
    app = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock(), send_photo=AsyncMock()))
    def unavailable(*args):
        raise TimeoutError("simulated HTTP timeout")
    monkeypatch.setattr("telegram_bot.bot.fetch_snapshot", unavailable)
    value = {"event": "recognition", "request_id": "session", "person": "son", "confidence": .94,
             "device_id": "test", "inference_ms": 30, "image": {"id": "a"*32}}
    asyncio.run(b.notify_event(app, "result", value))
    assert app.bot.send_message.await_count == 2
    assert "Recognition successful" in app.bot.send_message.call_args_list[0].args[1]
    app.bot.send_photo.assert_not_called()
    b.dispatch.assert_awaited_once()


def test_unknown_never_downloads_or_sends_photo(monkeypatch, tmp_path):
    b = bot(monkeypatch, tmp_path)
    b.cfg = replace(b.cfg, recognition_photo_enabled=True)
    b.dispatch = AsyncMock()
    fetch = AsyncMock()
    monkeypatch.setattr("telegram_bot.bot.fetch_snapshot", fetch)
    app = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock(), send_photo=AsyncMock()))
    asyncio.run(b.notify_event(app, "result", {"event": "recognition", "person": "Unknown"}))
    app.bot.send_message.assert_not_called()
    app.bot.send_photo.assert_not_called()
    b.dispatch.assert_not_called()
    fetch.assert_not_called()


def test_start_registers_recipient_before_command_can_produce_result(monkeypatch, tmp_path):
    b = bot(monkeypatch, tmp_path)
    async def dispatch(command):
        assert b.recognition_chats[command["request_id"]] == 123
        return {"ok": True}
    b.dispatch = dispatch
    message = SimpleNamespace(chat_id=123, reply_text=AsyncMock())
    asyncio.run(b.execute(message, {"action": "start", "request_id": "session"}))


def test_telegram_photo_failure_keeps_success_text_and_reports_delivery_problem(monkeypatch, tmp_path):
    from telegram.error import TelegramError
    b = bot(monkeypatch, tmp_path)
    b.cfg = replace(b.cfg, recognition_photo_enabled=True)
    b.recognition_chats["session"] = 123
    b.dispatch = AsyncMock(return_value={"ok": True, "state": "idle"})
    monkeypatch.setattr("telegram_bot.bot.fetch_snapshot", lambda *args: b"jpeg-bytes")
    app = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock(),
        send_photo=AsyncMock(side_effect=TelegramError("simulated upload failure"))))
    value = {"event": "recognition", "request_id": "session", "person": "son", "confidence": .94,
             "device_id": "test", "inference_ms": 30, "image": {"id": "a"*32}}
    asyncio.run(b.notify_event(app, "result", value))
    assert app.bot.send_message.await_count == 2
    assert "Recognition successful" in app.bot.send_message.call_args_list[0].args[1]
    assert "Không gửi được ảnh" in app.bot.send_message.call_args_list[1].args[1]
