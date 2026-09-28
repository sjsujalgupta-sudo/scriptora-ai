"""FastAPI application factory and entrypoint."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .api.routes import router
from .config import Settings, get_settings

UI_DIR = Path(__file__).parent / "ui"

logger = logging.getLogger("scriptora")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="Scriptora",
        description="Context-aware, voice-controlled AI subtitle agent powered by AssemblyAI",
        version="0.1.0",
    )
    app.state.settings = settings

    app.include_router(router)
    app.mount("/static", StaticFiles(directory=UI_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=UI_DIR / "templates")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        # `public_summary()` carries no credential material, so it is safe to
        # render into the page.
        return templates.TemplateResponse(
            request,
            "index.html",
            {"config": settings.public_summary()},
        )

    if not settings.has_api_key:
        logger.warning("ASSEMBLYAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    else:
        logger.info(
            "Scriptora starting with model=%s corrector=%s",
            settings.speech_model,
            settings.corrector_backend,
        )

    return app


app = create_app()


def run() -> None:
    """Console-script entrypoint (`scriptora`)."""
    import uvicorn

    settings = get_settings()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    uvicorn.run(
        "scriptora.main:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )
