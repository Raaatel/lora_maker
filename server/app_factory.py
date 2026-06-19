"""FastAPI application factory for LoRA Maker."""

from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from server.database import init_db
from server.services.training_manager import TrainingManager
from server.services.websocket_manager import ConnectionManager
from server.routers import pages, api_projects, api_upload, api_checkpoints, api_vastai, ws, api_validation

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"


def create_app() -> FastAPI:
    app = FastAPI(title="LoRA Maker", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "static").mkdir(exist_ok=True)
    (BASE_DIR / "templates").mkdir(exist_ok=True)

    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
    app.mount("/files", StaticFiles(directory=str(JOBS_DIR)), name="files")

    app.state.templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

    app.include_router(pages.router)
    app.include_router(api_projects.router)
    app.include_router(api_upload.router)
    app.include_router(api_checkpoints.router)
    app.include_router(api_vastai.router)
    app.include_router(ws.router)
    app.include_router(api_validation.router)

    @app.on_event("startup")
    async def on_startup():
        await init_db()
        app.state.ws_manager = ConnectionManager()
        app.state.training_manager = TrainingManager(app.state.ws_manager)

    @app.on_event("shutdown")
    async def on_shutdown():
        manager = getattr(app.state, "training_manager", None)
        if manager:
            await manager.cancel_all()

    return app
