"""HTTP and WebSocket routes.

The audio WebSocket is the whole product: the browser opens it, streams PCM
frames as binary messages, and receives JSON `ServerEvent`s as text. One
socket, one port, no separate service.
"""

from __future__ import annotations

import json
import logging
from contextlib import suppress

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from ..config import Settings
from ..models.correction import CorrectionResult
from ..services.session import ScriptoraSession

logger = logging.getLogger(__name__)

router = APIRouter()

# Sent as the first text frame so the browser can render configuration and any
# startup problem before it asks for the microphone.
HELLO_TYPE = "hello"


async def _send_json(websocket: WebSocket, payload: dict) -> None:
    await websocket.send_text(json.dumps(payload, default=str))


@router.get("/api/health")
async def health(request: Request) -> JSONResponse:
    """Liveness plus a credential-free summary of the configuration."""
    settings: Settings = request.app.state.settings
    return JSONResponse({"status": "ok", "config": settings.public_summary()})


@router.websocket("/ws/audio")
async def audio_socket(websocket: WebSocket) -> None:
    """The live session socket: audio up, events down."""
    await websocket.accept()
    settings: Settings = websocket.app.state.settings

    if not settings.has_api_key:
        await _send_json(
            websocket,
            {
                "type": "error",
                "message": (
                    "ASSEMBLYAI_API_KEY is not configured. Copy .env.example to .env, "
                    "add your key, and restart Scriptora."
                ),
                "payload": {"fatal": True},
            },
        )
        await websocket.close(code=4401)
        return

    async def send(event: dict) -> None:
        await _send_json(websocket, event)

    session = ScriptoraSession(settings, send)

    try:
        await _send_json(
            websocket,
            {
                "type": HELLO_TYPE,
                "message": "Scriptora ready.",
                "payload": {
                    "session_id": session.id,
                    "config": settings.public_summary(),
                    "vocabulary": session.context.terms,
                },
            },
        )

        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if (data := message.get("bytes")) is not None:
                await session.send_audio(data)
                continue

            if (text := message.get("text")) is not None:
                await _handle_control(session, text)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("Audio socket failed")
        with suppress(Exception):
            await _send_json(
                websocket,
                {
                    "type": "error",
                    "message": f"Scriptora hit an unexpected error: {type(exc).__name__}.",
                    "payload": {"fatal": True},
                },
            )
    finally:
        await session.stop()


async def _handle_control(session: ScriptoraSession, raw: str) -> None:
    """Handle a text control frame from the browser."""
    try:
        message = json.loads(raw)
    except json.JSONDecodeError:
        logger.debug("Ignoring non-JSON control frame")
        return

    action = message.get("action")

    if action == "start":
        await session.start()
    elif action == "stop":
        await session.stop()
    elif action == "command":
        command = (message.get("text") or "").strip()
        if not command:
            await session.emit_error("No command text was provided.")
            return
        result: CorrectionResult = await session.handle_command(command)
        logger.info("Command %r -> %s via %s", command, result.outcome.value, result.backend)
    elif action == "add_term":
        term = (message.get("term") or "").strip()
        if not term:
            await session.emit_error("No term was provided.")
            return
        await session.add_term(term)
    elif action == "clear":
        session.subtitles.clear()
        await session.broadcast_transcript()
    else:
        logger.debug("Ignoring unknown control action %r", action)
