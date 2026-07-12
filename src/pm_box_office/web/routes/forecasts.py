from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

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
                "include_coverage_fallback": request.query_params.get("include_coverage_fallback"),
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


def forecast_detail_context(request: Request, release_run_id: int) -> dict[str, object]:
    model_version = request.query_params.get("model_version")
    grid_variant = request.query_params.get("grid")
    conn = connect_database()
    try:
        detail = forecast_service.movie_timeline(
            conn,
            release_run_id=release_run_id,
            model_version=model_version,
            grid_variant=grid_variant,
        )
        conn.rollback()
    finally:
        conn.close()
    return {"request": request, "release_run_id": release_run_id, **detail}


@router.get("/forecasts/{release_run_id}/live")
def forecast_detail_live(request: Request, release_run_id: int) -> object:
    return templates.TemplateResponse(
        name="_forecast_detail_live.html",
        context=forecast_detail_context(request, release_run_id),
        request=request,
    )


@router.get("/forecasts/{release_run_id}/events")
async def forecast_events(request: Request, release_run_id: int) -> StreamingResponse:
    """Stream committed forecast-projection updates for one release.

    The outbox is polled rather than holding a database LISTEN connection, so
    the endpoint works identically in local single-process and multi-worker
    deployments. The detail page retains HTMX polling as a fallback.
    """
    model_version = request.query_params.get("model_version") or None
    try:
        last_id = max(0, int(request.headers.get("last-event-id") or request.query_params.get("after") or 0))
    except ValueError:
        last_id = 0

    async def stream() -> object:
        nonlocal last_id
        while True:
            if await request.is_disconnected():
                return
            conn = None
            try:
                from models.boxoffice.ui_projection import ui_events_after

                conn = connect_database()
                events = ui_events_after(
                    conn,
                    release_run_id=release_run_id,
                    model_version=model_version,
                    after_id=last_id,
                )
                conn.rollback()
                for event in events:
                    last_id = int(event["outbox_id"])
                    payload = json.dumps(event["payload"], default=str, separators=(",", ":"))
                    yield f"id: {last_id}\nevent: {event['event_type']}\ndata: {payload}\n\n"
            except Exception as exc:  # Keep a transient DB error from killing the browser subscription.
                yield f"event: pipeline_error\ndata: {json.dumps({'detail': type(exc).__name__})}\n\n"
            finally:
                if conn is not None:
                    conn.close()
            yield ": keepalive\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/forecasts/{release_run_id}")
def forecast_detail(request: Request, release_run_id: int) -> object:
    return templates.TemplateResponse(
        name="forecast_detail.html",
        context=forecast_detail_context(request, release_run_id),
        request=request,
    )
