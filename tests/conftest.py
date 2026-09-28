"""Shared fixtures. Every test runs with a fake key and no network."""

from __future__ import annotations

import pytest

from scriptora.config import Settings
from scriptora.services.context_service import ContextService
from scriptora.services.subtitle_service import SubtitleService

# Deliberately not a real credential.
FAKE_KEY = "test-key-not-a-credential"


@pytest.fixture
def settings() -> Settings:
    """Settings with `corrector_backend="rules"` so tests never hit the network."""
    return Settings(assemblyai_api_key=FAKE_KEY, corrector_backend="rules")


@pytest.fixture
def subtitles() -> SubtitleService:
    return SubtitleService()


@pytest.fixture
def context() -> ContextService:
    return ContextService()


@pytest.fixture
def context_with_fastapi() -> ContextService:
    """Context that has been told about FastAPI, as the demo user does by voice.

    FastAPI is not a seed term, so any test that exercises vocabulary-aware
    correction has to add it explicitly, exactly as a user would.
    """
    service = ContextService()
    service.add_term("FastAPI", source="remember")
    return service
