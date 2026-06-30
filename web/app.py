from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from web.routes import dashboard, runs


app = FastAPI(title="AMC Collection Control Panel")
app.mount("/static", StaticFiles(directory="web/static"), name="static")
app.include_router(dashboard.router)
app.include_router(runs.router)
