"""AssemblyAI Real-time Speech-to-Text integration.

This is the transcription foundation. Browser PCM frames are forwarded into
`AsyncRealTimeTranscriber`, and Turn events become subtitle updates.

Two behaviours are load-bearing and were established by testing against the
live API rather than by reading documentation:

1. `RealTimeTranscriberOptions.connect_timeout` defaults to 1.0 s, which fails
   on an ordinary broadband handshake. Scriptora raises it to 30 s.
2. AssemblyAI rejects any audio frame outside 50-1000 ms with error 3007.
   `ScriptoraSession` therefore resizes incoming frames before forwarding.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterable, Awaitable, Callable

from assemblyai.streaming.v3 import (
    AsyncRealTimeTranscriber,
    RealTimeError,
    RealTimeEvents,
    RealTimeParameters,
    RealTimeSessionParameters,
    RealTimeTranscriberOptions,
    SpeechModel,
    TurnEvent,
)

from ..config import Settings

logger = logging.getLogger(__name__)

# Callbacks ScriptoraSession subscribes to.
OnPartial = Callable[[str, int | None, float | None], Awaitable[None]]
OnFinal = Callable[[str, int | None, float | None], Awaitable[None]]
OnSpeechStart = Callable[[], Awaitable[None]]
OnStatus = Callable[[str, str], Awaitable[None]]
OnError = Callable[[str, bool], Awaitable[None]]

_API_HOST = "streaming.assemblyai.com"

# Error codes Scriptora can translate into an actionable message.
_ERROR_HINTS = {
    1001: "Invalid AssemblyAI API key. Check ASSEMBLYAI_API_KEY in your .env file.",
    1002: "AssemblyAI rejected this key. Check ASSEMBLYAI_API_KEY in your .env file.",
    # 1006 is the standard WebSocket "abnormal closure" code: the peer dropped
    # the TCP connection without sending a close frame, so there is no reason
    # text to relay. It is a transport failure, never a credential problem.
    1006: (
        "The AssemblyAI connection was dropped unexpectedly. Check your network, "
        "then press Start Listening to reconnect."
    ),
    3007: "AssemblyAI rejected an audio frame. Frames must be 50-1000 ms.",
    4001: "Too many streaming sessions. Wait a moment and try again.",
    4010: "This AssemblyAI key is not authorised for real-time streaming.",
}


def _options(api_key: str) -> RealTimeTranscriberOptions:
    return RealTimeTranscriberOptions(
        api_key=api_key,
        api_host=_API_HOST,
        connect_timeout=30.0,
        max_connection_retries=2,
        connection_retry_delay=0.5,
        # Allow AssemblyAI to flush the final turn when terminating.
        terminate_timeout=10.0,
    )


class AssemblyAIRealtimeService:
    """Owns one AssemblyAI real-time streaming session."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncRealTimeTranscriber | None = None
        self._session_id: str | None = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def session_id(self) -> str | None:
        return self._session_id

    # ----------------------------------------------------------- parameters
    def _build_parameters(self, keyterms: list[str] | None) -> RealTimeParameters:
        valid_models = {model.value for model in SpeechModel}
        speech_model = (
            SpeechModel(self._settings.speech_model)
            if self._settings.speech_model in valid_models
            else SpeechModel.universal_3_5_pro
        )
        params = RealTimeParameters(
            speech_model=speech_model,
            sample_rate=self._settings.sample_rate,
            format_turns=True,
            mode=self._settings.stream_mode,
            min_turn_silence=self._settings.min_turn_silence,
            max_turn_silence=self._settings.max_turn_silence,
        )
        if keyterms:
            # `keyterms_prompt` is typed list[str] by the SDK. It biases
            # AssemblyAI's own transcription, so terms are heard correctly the
            # first time instead of being corrected afterwards.
            params.keyterms_prompt = list(keyterms)
        return params

    # ------------------------------------------------------------- handlers
    def _register(
        self,
        client: AsyncRealTimeTranscriber,
        *,
        on_partial: OnPartial,
        on_final: OnFinal,
        on_speech_start: OnSpeechStart,
        on_status: OnStatus,
        on_error: OnError,
    ) -> None:
        client.on(RealTimeEvents.Begin, self._on_begin(on_status))
        client.on(RealTimeEvents.Turn, self._on_turn(on_partial, on_final))
        client.on(RealTimeEvents.SpeechStarted, self._on_speech_started(on_speech_start))
        client.on(RealTimeEvents.Error, self._on_error(on_error))
        client.on(RealTimeEvents.Warning, self._on_warning)
        client.on(RealTimeEvents.Termination, self._on_termination)

    def _on_begin(self, on_status: OnStatus):
        async def handler(_client, event) -> None:
            self._session_id = event.id
            self._connected = True
            logger.info("AssemblyAI session %s opened", event.id)
            await on_status("listening", "Connected to AssemblyAI. Start speaking.")

        return handler

    def _on_turn(self, on_partial: OnPartial, on_final: OnFinal):
        async def handler(_client, event: TurnEvent) -> None:
            text = (event.transcript or "").strip()
            if not text:
                # Empty transcripts are normal during silence; never surface them.
                return
            confidence = getattr(event, "end_of_turn_confidence", None)
            if event.end_of_turn:
                await on_final(text, event.turn_order, confidence)
            else:
                await on_partial(text, event.turn_order, confidence)

        return handler

    def _on_speech_started(self, on_speech_start: OnSpeechStart):
        async def handler(_client, _event) -> None:
            await on_speech_start()

        return handler

    def _on_error(self, on_error: OnError):
        async def handler(_client, error: RealTimeError) -> None:
            code = getattr(error, "code", None)
            friendly = _ERROR_HINTS.get(code)
            if friendly is None:
                # Log the raw text, but never show credential-shaped content.
                logger.warning("AssemblyAI error code=%s: %s", code, str(error)[:300])
                friendly = f"AssemblyAI reported an error (code {code}). See the server log."
            self._connected = False
            await on_error(friendly, True)

        return handler

    async def _on_warning(self, _client, event) -> None:
        logger.warning(
            "AssemblyAI warning %s: %s",
            getattr(event, "warning_code", None),
            getattr(event, "warning", ""),
        )

    async def _on_termination(self, _client, event) -> None:
        logger.info(
            "AssemblyAI session ended after %ss of audio",
            getattr(event, "audio_duration_seconds", None),
        )

    # ------------------------------------------------------------ lifecycle
    async def connect(
        self,
        keyterms_prompt: str | None,
        *,
        on_partial: OnPartial,
        on_final: OnFinal,
        on_speech_start: OnSpeechStart,
        on_status: OnStatus,
        on_error: OnError,
    ) -> None:
        """Open the streaming session. Raises on auth or network failure."""
        client = AsyncRealTimeTranscriber(_options(self._settings.require_api_key()))
        self._register(
            client,
            on_partial=on_partial,
            on_final=on_final,
            on_speech_start=on_speech_start,
            on_status=on_status,
            on_error=on_error,
        )
        try:
            await client.connect(self._build_parameters(keyterms_prompt))
        except Exception:
            self._client = None
            self._connected = False
            raise
        self._client = client

    async def stream(self, frames: AsyncIterable[bytes]) -> None:
        """Drain an async iterable of PCM frames into the live session.

        The SDK's `stream` is one-shot: it consumes the iterable and then
        returns, so this must be awaited exactly once per session.

        Deliberately not gated on `_connected`. The `Begin` event is delivered
        asynchronously, so a task started right after `connect()` would
        otherwise see `False` and drop every frame. The SDK's own
        `_ensure_connected` already guards genuine pre-connect misuse.
        """
        if self._client is None:
            return
        await self._client.stream(frames)

    async def update_keyterms(self, keyterms: list[str]) -> bool:
        """Push updated vocabulary into a live session.

        Returns True when AssemblyAI accepted the update. This is what makes
        the "Remember FastAPI as a technical term" demo land: the *next*
        utterance is transcribed correctly by AssemblyAI itself, before any
        correction is required.
        """
        if self._client is None or not self._connected or not keyterms:
            return False
        try:
            # `set_params` is a coroutine on the async client. Calling it
            # without await sent nothing while still reporting success, so the
            # demo claimed the vocabulary update had landed when it had not.
            await self._client.set_params(RealTimeSessionParameters(keyterms_prompt=list(keyterms)))
            return True
        except Exception as exc:
            logger.info("Could not update keyterms mid-session: %s", type(exc).__name__)
            return False

    async def disconnect(self) -> None:
        """Close the session, always terminating so billing stops."""
        self._connected = False
        client, self._client = self._client, None
        if client is None:
            return
        try:
            await client.disconnect(terminate=True)
        except Exception as exc:
            # WARNING, not INFO: if the termination frame never got through, the
            # final turn was not flushed and the last words of the session are
            # lost. The Stop summary still reports a clean stop, so this log line
            # is the only evidence that it was not.
            logger.warning("AssemblyAI session did not terminate cleanly: %s", type(exc).__name__)
