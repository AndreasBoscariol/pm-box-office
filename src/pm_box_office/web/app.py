from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from pm_box_office.orchestration import scheduler
from pm_box_office.web.routes import amc, home, runs, sources


WEB_ROOT = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    autorun_task = asyncio.create_task(scheduler.run_forever())
    try:
        yield
    finally:
        await scheduler.shutdown_task(autorun_task)


app = FastAPI(title="Box Office Ingest Console", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(WEB_ROOT / "static")), name="static")
app.include_router(home.router)
app.include_router(amc.router)
app.include_router(runs.router)
app.include_router(sources.router)
