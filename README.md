# PM Box Office

Box office ingest and collection workspace for:

- The Numbers domestic box-office actuals
- Boxoffice Pro forecast article/range ingestion
- Audience snapshots from IMDb, Letterboxd, and Wikidata-assisted matching
- Wikipedia pageview and revision activity
- Rotten Tomatoes critic/review snapshots
- AMC theatres, showtimes, sampled seat snapshots, and collection operations
- Polymarket account scans and file reports

The Python package lives under `src/pm_box_office`. Use `python -m pm_box_office...`
or the console scripts declared in `pyproject.toml`.

## Data

Postgres is the system of record for DB-backed ingest and AMC collection state.
Raw HTTP caches live under `data/raw/`. Polymarket account scans write local
CSV/HTML/cache outputs under `data/raw/polymarket/` and `results/`.

DB-backed commands resolve the database URL in this order:

1. `--database-url`
2. `DATABASE_URL`
3. `POSTGRES_DSN`
4. `.env` in the repo root

Example `.env`:

```sh
DATABASE_URL=postgresql://localhost/pm_box_office
```

## Setup

```sh
cd /Users/andreasboscariol/Desktop/PolyMarket/pm-box-office
.venv/bin/pip install -e .
# Only needed for historical/research notebooks and diagnostics:
.venv/bin/pip install -e '.[research]'
```

Run migrations:

```sh
.venv/bin/alembic upgrade head
```

## Ingest Commands

The Numbers actuals:

```sh
.venv/bin/python -m pm_box_office.sources.the_numbers.ingest --dry-run
.venv/bin/python -m pm_box_office.sources.the_numbers.ingest --start-date 2026-06-01 --end-date 2026-06-30
```

Boxoffice Pro:

```sh
.venv/bin/python -m pm_box_office.sources.boxofficepro.ingest --dry-run
.venv/bin/python -m pm_box_office.sources.boxofficepro.ingest --start-date 2026-06-01 --end-date 2026-06-30
```

Box Office Theory Substack:

```sh
.venv/bin/python -m pm_box_office.sources.boxofficetheory_substack.ingest --dry-run
.venv/bin/python -m pm_box_office.sources.boxofficetheory_substack.ingest --start-date 2026-06-01 --end-date 2026-06-30
.venv/bin/python -m pm_box_office.sources.boxofficetheory_substack.ingest --full-refresh
```

Edward Douglas Substack:

```sh
.venv/bin/python -m pm_box_office.sources.edwarddouglas_substack.ingest --dry-run
.venv/bin/python -m pm_box_office.sources.edwarddouglas_substack.ingest --start-date 2026-06-01 --end-date 2026-06-30
.venv/bin/python -m pm_box_office.sources.edwarddouglas_substack.ingest --full-refresh
```

Audience snapshots:

```sh
.venv/bin/python -m pm_box_office.sources.audience.ingest --dry-run
.venv/bin/python -m pm_box_office.sources.audience.ingest --max-movies 10
```

Wikipedia:

```sh
.venv/bin/python -m pm_box_office.sources.wikipedia.ingest --dry-run
.venv/bin/python -m pm_box_office.sources.wikipedia.ingest
```

Rotten Tomatoes:

```sh
.venv/bin/python -m pm_box_office.sources.rotten_tomatoes.ingest --dry-run
.venv/bin/python -m pm_box_office.sources.rotten_tomatoes.ingest --max-movies 10
```

AMC collection:

```sh
.venv/bin/python -m pm_box_office.sources.amc.collect --help
.venv/bin/python -m pm_box_office.sources.amc.jobs.worker --help
.venv/bin/python -m pm_box_office.sources.amc.collect init-db
```

Live forecast refreshes:

```sh
.venv/bin/python -m models.boxoffice.refresh_worker --help
.venv/bin/python -m models.boxoffice.refresh_worker
```

AMC seat snapshots enqueue debounced forecast refreshes by default. Set
`AMC_FORECAST_REFRESH_ENABLED=0` to disable enqueueing, or
`AMC_FORECAST_REFRESH_DEBOUNCE_SECONDS=20` and `FORECAST_REFRESH_MODEL_VERSION=latest`
to tune refresh behavior. Starting sampled seat collection schedules every
movie in the live AMC inventory; the movie checkboxes are informational and do
not gate seat collection.

Polymarket accounts:

```sh
.venv/bin/python -m pm_box_office.sources.polymarket.accounts --help
```

## Web Interface

The FastAPI web interface is an ingest operations console with source
orchestration as the entry point and AMC collection under its own section.

```sh
.venv/bin/uvicorn pm_box_office.web.app:app --reload
```

Open:

- `http://127.0.0.1:8000/` or `/sources` for ingest source orchestration
- `http://127.0.0.1:8000/amc` for AMC campaign collection

The forecast refresh worker reads the explicitly selected artifact version from
`models/boxoffice/ACTIVE_MODEL`. Historical research and diagnostic runners are
not part of the operational workflow.

## Model Documentation

The production opening-weekend model is documented in
[`docs/production_model.md`](docs/production_model.md), including the
pre-release policy, live daily composition, opening-Thursday and preview prior
updates, AMC same-day plug-in behavior, simulation intervals, and emission
idempotency rules.

## Tests

```sh
.venv/bin/python -m pytest tests/db tests/sources tests/orchestration tests/web
```
