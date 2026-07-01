# PM Box Office

Box office prediction workspace for collecting actuals from The Numbers,
building Wikipedia and AMC signals, training same-day box office models, and
running supporting Polymarket account scans.

The real Python code lives under `src/pm_box_office`. Legacy Python entrypoints
under `scripts/` and root `web/` have been removed; use `python -m
pm_box_office...` or the console scripts declared in `pyproject.toml`.

## Data Flow

- The Numbers is the canonical source for daily box office actuals.
- Wikipedia ingests pageview and revision activity for movies already present
  in the The Numbers tables.
- AMC collects theatres, showtimes, seat snapshots, and derived same-day
  prediction features.
- Model training reads from Postgres and writes artifacts under `results/`.
- Polymarket account scanning writes CSV/HTML files only; it does not write to
  Postgres yet.

Raw HTTP caches live under `data/raw/`, durable generated data under
`data/processed/`, and reports/models under `results/`.

## Setup

```sh
cd /Users/andreasboscariol/Desktop/PolyMarket/pm-box-office
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

Postgres connection precedence for DB-backed commands:

1. `--database-url` when provided
2. `DATABASE_URL`
3. `POSTGRES_DSN`
4. `.env` in the repo root

Example `.env`:

```sh
DATABASE_URL=postgresql://localhost/pm_box_office
```

## Canonical Commands

The Numbers actuals:

```sh
.venv/bin/python -m pm_box_office.sources.the_numbers.ingest --dry-run
.venv/bin/python -m pm_box_office.sources.the_numbers.ingest --start-date 2026-06-01 --end-date 2026-06-30
```

Boxoffice Pro forecast articles, matched back to The Numbers movies when possible:

```sh
.venv/bin/python -m pm_box_office.sources.boxofficepro.ingest --dry-run
.venv/bin/python -m pm_box_office.sources.boxofficepro.ingest --start-date 2026-06-01 --end-date 2026-06-30
```

Boxoffice Pro may return HTTP 403 to plain Python requests. When that happens,
seed the raw HTML cache from a normal browser and run offline:

```sh
.venv/bin/python -m pm_box_office.sources.boxofficepro.ingest --print-cache-paths --max-pages 1
# Save the browser page source for each URL to its printed cache path.
.venv/bin/python -m pm_box_office.sources.boxofficepro.ingest --offline --start-date 2026-06-01 --end-date 2026-06-30
```

If an article page is missing from cache, the offline error prints the exact
article URL and cache path to save next.

Wikipedia features, after The Numbers has populated movies/releases/actuals:

```sh
.venv/bin/python -m pm_box_office.sources.wikipedia.ingest --dry-run
.venv/bin/python -m pm_box_office.sources.wikipedia.ingest --movie-limit 10
```

AMC control CLI:

```sh
.venv/bin/python -m pm_box_office.sources.amc.collect init-db
.venv/bin/python -m pm_box_office.sources.amc.collect ingest-theatres
.venv/bin/python -m pm_box_office.sources.amc.collect create-inventory-run 2026-06-30
.venv/bin/python -m pm_box_office.sources.amc.collect create-seat-run 2026-06-30
```

AMC worker:

```sh
.venv/bin/python -m pm_box_office.sources.amc.jobs.worker --verbose
```

Train AMC box office models:

```sh
.venv/bin/python -m pm_box_office.models.train
```

Find Polymarket movie/box-office accounts:

```sh
.venv/bin/python -m pm_box_office.sources.polymarket.accounts
```

The only remaining `scripts/` files are shell shortcuts for common ingest smoke
tests:

```sh
scripts/run_scrape_the_numbers.sh --dry-run
scripts/run_ingest_wikipedia_boxoffice.sh --dry-run
```

## Web App

Run locally:

```sh
.venv/bin/uvicorn pm_box_office.web.app:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

The web app initializes AMC tables, syncs theatres, creates collection runs,
selects movies, starts seat collection, and can start a local AMC worker.
By default, web-created seat collection runs collect one seat snapshot per
showtime at 5 minutes before start.

Local worker controls:

```sh
AMC_LOCAL_WORKER_COUNT=3 AMC_WORKER_BATCH_LIMIT=1 AMC_WORKER_DELAY_SECONDS=1.0 \
  .venv/bin/uvicorn pm_box_office.web.app:app --reload --host 127.0.0.1 --port 8000
```

Use more workers when the dashboard shows due or late backlog. Each worker
claims distinct queue rows from Postgres.

Run with Docker:

```sh
docker compose up --build
docker compose up --build --scale worker=3
```

Docker starts Postgres, the web app, and an AMC worker.

## Tests

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover
```

Postgres integration tests skip unless `TEST_DATABASE_URL` or `DATABASE_URL` is
available. Use `TEST_DATABASE_URL` when you want tests to avoid your dev
database.
