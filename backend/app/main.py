"""FastAPI entry point.

Run:  uvicorn app.main:app --reload --port 8000
Then: http://localhost:8000        (dev console)
      http://localhost:8000/docs   (interactive API docs)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router, ws_router
from .config import settings
from .runtime.hub import Hub
from .runtime.lab import LabManager
from .runtime.session import SimulationSession
from .store.db import RunStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub = Hub()
    store = RunStore(settings.db_path, enabled=settings.persist)
    app.state.hub = hub
    app.state.store = store
    app.state.session = SimulationSession(hub, store, settings)
    app.state.lab = LabManager(hub, store, settings.max_lab_runs)
    await app.state.session.reset()  # ready to go: default scenario, paused at 00:00
    yield
    await app.state.session.shutdown()


app = FastAPI(
    title="Renewable Energy Orchestrator",
    version="0.1.0",
    description="Simulator + agent runtime for the ET AI Hackathon (Problem 4). "
                "WebSocket stream at /ws; dev console at /.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"],
                   allow_headers=["*"])
app.include_router(router)
app.include_router(ws_router)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


DASHBOARD = STATIC / "dashboard"
if DASHBOARD.exists():  # React dashboard (built from /frontend)
    app.mount("/dashboard", StaticFiles(directory=DASHBOARD, html=True), name="dashboard")


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def home():
    if (DASHBOARD / "index.html").exists():
        return RedirectResponse("/dashboard/")
    return FileResponse(STATIC / "index.html")


@app.get("/console", include_in_schema=False)
async def console():
    return FileResponse(STATIC / "index.html")
