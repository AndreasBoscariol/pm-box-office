from __future__ import annotations

from pathlib import Path

from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape

from pm_box_office.web.time_format import duration_until, local_clock_time, local_short_datetime, time_ago


WEB_ROOT = Path(__file__).resolve().parent


templates = Jinja2Templates(
    env=Environment(
        loader=FileSystemLoader(str(WEB_ROOT / "templates")),
        autoescape=select_autoescape(("html", "xml")),
        cache_size=0,
    )
)
templates.env.filters["duration_until"] = duration_until
templates.env.filters["local_short_datetime"] = local_short_datetime
templates.env.filters["local_time"] = local_clock_time
templates.env.filters["time_ago"] = time_ago
