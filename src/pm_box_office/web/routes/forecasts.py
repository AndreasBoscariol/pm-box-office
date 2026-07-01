from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape

from pm_box_office.db.connection import connect_database
from pm_box_office.web.services import forecast_service


WEB_ROOT = Path(__file__).resolve().parents[1]
router = APIRouter()
templates = Jinja2Templates(
    env=Environment(
        loader=FileSystemLoader(str(WEB_ROOT / "templates")),
        autoescape=select_autoescape(("html", "xml")),
        cache_size=0,
    )
)


@router.get("/forecasts")
def forecasts_page(request: Request, movie_id: int | None = None) -> object:
    return templates.TemplateResponse(
        name="forecasts.html",
        context={
            "request": request,
            "selected_movie_id": movie_id or "",
        },
        request=request,
    )


@router.get("/forecasts/movies")
def search_movies(q: str = Query(default="", max_length=120), limit: int = Query(default=12, ge=1, le=50)) -> object:
    conn = connect_database()
    try:
        movies = forecast_service.search_forecast_movies(conn, query=q, limit=limit)
        conn.rollback()
    finally:
        conn.close()
    return JSONResponse({"movies": movies})


@router.get("/forecasts/movies/{movie_id}")
def movie_forecast(movie_id: int) -> object:
    conn = connect_database()
    try:
        forecast = forecast_service.forecast_movie(conn, movie_id=movie_id)
        conn.rollback()
    finally:
        conn.close()
    if forecast is None:
        raise HTTPException(status_code=404, detail="Movie forecast not found")
    forecast["feature_availability"] = forecast_service.feature_availability_summary(forecast["snapshots"])
    return JSONResponse(forecast)
