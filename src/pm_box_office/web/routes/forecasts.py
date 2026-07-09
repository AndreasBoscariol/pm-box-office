from __future__ import annotations

from fastapi import APIRouter, Request

from pm_box_office.db.connection import connect_database
from pm_box_office.web.services import forecasts as forecast_service
from pm_box_office.web.templating import templates


router = APIRouter()


@router.get("/forecasts")
def forecast_dashboard(request: Request) -> object:
    conn = connect_database()
    try:
        default_model_version = forecast_service.latest_model_version(conn)
        filters = forecast_service.normalize_filters(
            {
                "q": request.query_params.get("q"),
                "release_start": request.query_params.get("release_start"),
                "release_end": request.query_params.get("release_end"),
                "model_version": request.query_params.get("model_version"),
                "target": request.query_params.get("target"),
            },
            default_model_version=default_model_version,
        )
        dashboard = forecast_service.dashboard(conn, filters)
        conn.rollback()
    finally:
        conn.close()
    return templates.TemplateResponse(
        name="forecasts.html",
        context={"request": request, **dashboard},
        request=request,
    )


@router.get("/forecasts/{release_run_id}")
def forecast_detail(request: Request, release_run_id: int) -> object:
    model_version = request.query_params.get("model_version")
    conn = connect_database()
    try:
        detail = forecast_service.movie_timeline(
            conn,
            release_run_id=release_run_id,
            model_version=model_version,
        )
        conn.rollback()
    finally:
        conn.close()
    return templates.TemplateResponse(
        name="forecast_detail.html",
        context={"request": request, "release_run_id": release_run_id, **detail},
        request=request,
    )

