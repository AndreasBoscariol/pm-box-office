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
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
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

Forecast modeling UI and research/model training modules were removed on the
`cleanup-core-ingest-web` branch. A future forecast page should read from
database tables directly instead of importing research/model code.

## Tests

```sh
.venv/bin/python -m pytest tests/db tests/sources tests/orchestration tests/web
```
