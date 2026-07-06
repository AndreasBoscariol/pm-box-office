# Core Ingest/Web Cleanup Record

Branch: `cleanup-core-ingest-web`

Goal: keep the database, all ingest scripts, the Polymarket accounts script and
data, and the web interface. Remove exploratory forecast modeling, paper
recreation, research diagnostics, generated model/research artifacts, and the
model-backed forecast web page.

## Kept

- `src/pm_box_office/db/`
- `src/pm_box_office/domain/`
- `src/pm_box_office/sources/`
- `src/pm_box_office/orchestration/`
- `src/pm_box_office/web/`, now an ingest operations console with AMC collection isolated under its own section
- `src/pm_box_office/config.py`
- `alembic/`
- `scripts/run_scrape_the_numbers.sh`
- `scripts/run_ingest_wikipedia_boxoffice.sh`
- `Dockerfile`
- `docker-compose.yml`
- `alembic.ini`
- `.env.example`
- `docs/postgres_setup.md`
- Focused tests under `tests/db`, `tests/orchestration`, `tests/sources`, and `tests/web`

## Removed

- `src/pm_box_office/models/`
- `src/pm_box_office/research/`
- `src/pm_box_office/features/`
- `configs/`
- `docs/papers_ocr/`
- `tests/models/`
- `tests/research/`
- `tests/papers/`
- `tests/analytics/`
- `src/pm_box_office/web/routes/forecasts.py`
- `src/pm_box_office/web/services/forecast_service.py`
- `src/pm_box_office/web/templates/forecasts.html`
- Forecast-only CSS selectors from `src/pm_box_office/web/static/app.css`
- Forecast nav links from the dashboard and sources pages
- Forecast/model generated outputs under `results/models`, `results/papers`, and `results/research`
- Model/research console scripts and dependencies from `pyproject.toml` and `requirements.txt`

## Data

`data/` was left intact. It contains raw ingest caches, Polymarket account data,
AMC logs/run state, and other local assets that should be reviewed separately
before any deletion.

## Migration Note

Historical Alembic migration files were kept, including forecast-related
migrations. This preserves revision continuity for any local database that has
already run them. If the forecast tables should be dropped from the database
itself, add a new cleanup migration instead of deleting old migration files.

## Future Forecast UI

A future forecast page should be rebuilt as a DB-only read interface. It should
not import research scripts or training/model packages directly from the web
layer.
