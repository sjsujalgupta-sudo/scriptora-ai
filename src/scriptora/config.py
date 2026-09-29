"""Application configuration.

The AssemblyAI key is read from the environment only. It is never logged,
never serialised into an event payload, and never sent to the browser.
`Settings.public_summary()` exists specifically so the UI can be told
"credentials are configured" without learning the credential itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

# AssemblyAI's LLM Gateway is OpenAI-compatible and authenticated with the
# *same* AssemblyAI key, so Scriptora needs no second provider credential.
LLM_GATEWAY_URL = "https://llm-gateway.assemblyai.com/v1/chat/completions"

# This key can use this model (verified against the gateway). It does not
# support `response_format`, so correction output is prompt-constrained JSON
# that Pydantic then validates.
DEFAULT_LLM_MODEL = "qwen3.5-4b-32k-fast"

VALID_SPEECH_MODELS = frozenset(
    {
        "universal-3-5-pro",
        "universal-3-6",
        "universal-3-6-pro",
        "universal-streaming-english",
        "universal-streaming-multilingual",
        "whisper-rt",
    }
)
VALID_STREAM_MODES = frozenset({"min_latency", "balanced", "max_accuracy"})
VALID_CORRECTOR_BACKENDS = frozenset({"auto", "llm", "rules"})


class ConfigError(RuntimeError):
    """Raised for misconfiguration that the user can actually fix."""


@dataclass(frozen=True)
class Settings:
    assemblyai_api_key: str | None = field(default=None, repr=False)
    speech_model: str = "universal-3-5-pro"
    llm_model: str = DEFAULT_LLM_MODEL
    stream_mode: str = "balanced"
    corrector_backend: str = "auto"
    host: str = "127.0.0.1"
    port: int = 8000

    # Audio contract with AssemblyAI. Mono signed 16-bit little-endian PCM.
    # AssemblyAI rejects frames outside 50-1000 ms, so the browser is told to
    # send FRAME_DURATION_MS.
    sample_rate: int = 16000
    frame_duration_ms: int = 100

    # Turn detection. Subtitles want patience: a turn should close on a
    # natural pause, not mid-phrase.
    min_turn_silence: int = 320
    max_turn_silence: int = 1600

    # Lower bound on an interpreter's confidence before a spoken repair is
    # executed. Below it, or without one, the turn stays in the transcript as
    # ordinary speech.
    intent_confidence_threshold: float = 0.6

    @property
    def has_api_key(self) -> bool:
        return bool(self.assemblyai_api_key)

    @property
    def llm_enabled(self) -> bool:
        """`auto` and `llm` both attempt the gateway; `rules` never does."""
        return self.corrector_backend in ("auto", "llm") and self.has_api_key

    @property
    def frame_bytes(self) -> int:
        """Bytes per audio frame at the configured rate/duration."""
        samples = self.sample_rate * self.frame_duration_ms // 1000
        return samples * 2  # mono, signed 16-bit

    def require_api_key(self) -> str:
        if not self.assemblyai_api_key:
            raise ConfigError(
                "ASSEMBLYAI_API_KEY is not set. Copy .env.example to .env and add "
                "your key, then restart Scriptora."
            )
        return self.assemblyai_api_key

    def public_summary(self) -> dict[str, object]:
        """Safe-to-send-to-browser view. Contains no credential material."""
        return {
            "api_key_configured": self.has_api_key,
            "speech_model": self.speech_model,
            "stream_mode": self.stream_mode,
            "llm_model": self.llm_model if self.llm_enabled else None,
            "corrector_backend": self.corrector_backend,
            "sample_rate": self.sample_rate,
            "frame_duration_ms": self.frame_duration_ms,
            "frame_bytes": self.frame_bytes,
        }


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _int_env(name: str, default: int) -> int:
    raw = _clean(os.getenv(name))
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = _clean(os.getenv(name))
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(0.0, min(1.0, value))


def load_settings() -> Settings:
    """Build settings from the environment, falling back to safe defaults."""
    speech_model = _clean(os.getenv("SCRIPTORA_SPEECH_MODEL")) or "universal-3-5-pro"
    if speech_model not in VALID_SPEECH_MODELS:
        speech_model = "universal-3-5-pro"

    stream_mode = _clean(os.getenv("SCRIPTORA_STREAM_MODE")) or "balanced"
    if stream_mode not in VALID_STREAM_MODES:
        stream_mode = "balanced"

    backend = (_clean(os.getenv("SCRIPTORA_CORRECTOR_BACKEND")) or "auto").lower()
    if backend not in VALID_CORRECTOR_BACKENDS:
        backend = "auto"

    frame_duration_ms = _int_env("SCRIPTORA_FRAME_DURATION_MS", 100)
    # AssemblyAI hard-rejects frames outside 50-1000 ms.
    frame_duration_ms = max(50, min(1000, frame_duration_ms))

    return Settings(
        assemblyai_api_key=_clean(os.getenv("ASSEMBLYAI_API_KEY")),
        speech_model=speech_model,
        llm_model=_clean(os.getenv("SCRIPTORA_LLM_MODEL")) or DEFAULT_LLM_MODEL,
        stream_mode=stream_mode,
        corrector_backend=backend,
        host=_clean(os.getenv("SCRIPTORA_HOST")) or "127.0.0.1",
        port=_int_env("SCRIPTORA_PORT", 8000),
        frame_duration_ms=frame_duration_ms,
        intent_confidence_threshold=_float_env("SCRIPTORA_INTENT_CONFIDENCE", 0.6),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
