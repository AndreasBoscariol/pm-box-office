from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from pm_box_office.web.routes import dashboard, runs


WEB_ROOT = Path(__file__).resolve().parent

app = FastAPI(title="AMC Collection Control Panel")
app.mount("/static", StaticFiles(directory=str(WEB_ROOT / "static")), name="static")
app.include_router(dashboard.router)
app.include_router(runs.router)
