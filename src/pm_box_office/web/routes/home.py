from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import RedirectResponse


router = APIRouter()


@router.get("/")
def index() -> RedirectResponse:
    return RedirectResponse(url="/sources", status_code=303)
