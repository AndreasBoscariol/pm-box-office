# PM Box Office

Box office prediction workspace for collecting actuals from The Numbers,
building Wikipedia and AMC signals, training same-day box office models, and
running supporting Polymarket account scans.

The real Python code lives under `src/pm_box_office`. Legacy Python entrypoints
under `scripts/` and root `web/` have been removed; use `python -m
pm_box_office...` or the console scripts declared in `pyproject.toml`.

## Data Flow

- The Numbers is the canonical source for movies, release runs, and daily
  domestic box office actuals.
- Boxoffice Pro ingests Weekend Preview forecast ranges and links them back to
  The Numbers movies when a confident match exists.
- Audience snapshots add IMDb vote/rating counts and Letterboxd fan/rating
  counts for upcoming and recently active The Numbers movies.
- Wikipedia ingests pageview and revision activity for movies already present
  in the The Numbers tables.
- AMC collects theatres, showtimes, seat snapshots, and derived same-day
  prediction features.
- Model training reads from Postgres and writes artifacts under `results/`.
- Polymarket account scanning writes CSV/HTML files only; it does not write to
  Postgres yet.

Raw HTTP caches live under `data/raw/`, durable generated data under
`data/processed/`, and reports/models under `results/`.

## Ingest Scripts

All DB-backed ingest commands initialize the tables they need, write raw HTTP
responses to a cache directory under `data/raw/`, and use the Postgres URL
resolved from `--database-url`, `DATABASE_URL`, `POSTGRES_DSN`, or `.env`.
Networked runs are intentionally slow and cache-first; use `--offline` to
require cached pages and `--refresh` to refetch/reparse.

### The Numbers

Module: `pm_box_office.sources.the_numbers.ingest`

The Numbers is the base box-office actuals loader. For each date in the
requested range it reads the daily domestic chart page, discovers movie URLs,
then fetches each movie page and imports the full daily domestic run.

Collected data:

- Daily chart rows: rank, previous rank, gross, day/week changes, theatres,
  per-theatre gross, cumulative gross, and days in release.
- Movie identity: The Numbers movie URL, title, release year, and opusdata ID
  when present.
- Full movie daily run: box-office date, day number, rank, gross,
  percent-yesterday, percent-last-week, theatres, per-theatre gross,
  cumulative gross, and preview flag.
- Raw page provenance: source URL, cache path, fetched timestamp, and SHA-256.

Main database writes:

- `raw_source_pages` records cached chart/movie page provenance.
- `daily_chart_pages` stores each source daily chart row.
- `movies` is upserted by `movie_url` and becomes the shared movie dimension.
- `movie_source_ids` gets a `the_numbers` source ID when that cross-source
  table exists.
- `release_runs` gets one `US_CA` / `movie_page_full_run` row per movie page.
- `daily_box_office` stores canonical The Numbers actuals keyed by
  `(release_run_id, box_office_date, source)`.
- `box_office_import_issues` records reconciliation differences between chart
  rows and movie-page daily rows.

### Boxoffice Pro

Module: `pm_box_office.sources.boxofficepro.ingest`

Boxoffice Pro imports only high-confidence Weekend Preview forecast content.
It discovers articles from the forecasts/tracking RSS feed, falling back to
paginated archive pages when the requested start date is older than the feed
window. Article fetches use HTTP first and can fall back to Playwright when the
site blocks plain HTTP.

Collected data:

- Forecast article metadata: URL, title, author, discovered/published date,
  article type, parser version, fetch status, cache path, and SHA-256.
- Weekend prediction rows from "Boxoffice Podium" blocks: source movie title,
  distributor, release status, rank, forecast metric, low/high USD range,
  showtime market share when present, target weekend dates, raw forecast text,
  and parser context.
- Rejected or unavailable article details for parser review.

Main database writes:

- `boxofficepro_articles` stores discovered and parsed article provenance.
- `boxofficepro_weekend_predictions` stores forecast ranges and match metadata.
- `boxofficepro_movie_match_overrides` can pin a normalized source title or
  article-specific title to a The Numbers `movie_url`.
- `boxofficepro_ingest_issues` stores blocked pages, rejected blocks, and
  no-prediction parse outcomes.

Matching uses `movies` from The Numbers. Matched forecast rows populate
`matched_movie_id`; unmatched rows are still retained with `match_status` and
notes so they can be reviewed or fixed with an override.

### Audience Snapshots

Module: `pm_box_office.sources.audience.ingest`

Audience snapshots collect pre-release and current-release audience interest
signals for movies already known through The Numbers data. Candidate movies
come from the The Numbers release schedule plus recent box-office activity in
`daily_box_office`.

Collected data:

- The Numbers release schedule rows for upcoming releases.
- IMDb title metadata and official IMDb ratings/vote-count snapshots from
  `datasets.imdbws.com`.
- Wikidata-assisted external IDs for IMDb/TMDB/Letterboxd matching unless
  `--skip-wikidata` is used.
- Letterboxd aggregate film-page snapshots, currently fan count and average
  rating, plus parse status and source provenance.
- Per-movie ingest state so failed IMDb/Letterboxd stages can be retried.

Main database writes:

- `the_numbers_release_schedule` stores upcoming release candidates.
- `movies` is upserted from the release schedule and recent chart activity.
- `imdb_titles`, `movie_imdb_titles`, and `imdb_title_snapshots` store IMDb
  identity matches and dated rating/vote snapshots.
- `letterboxd_films`, `movie_letterboxd_films`, and
  `letterboxd_film_snapshots` store Letterboxd identity matches and dated
  snapshots.
- `audience_ingest_state` tracks source/stage status, attempts, and errors.
- `analytics.movie_audience_daily_features_v1` and
  `analytics.box_office_audience_panel_v1` expose joined audience and box
  office features for analysis/modeling.

Useful command:

```sh
.venv/bin/python -m pm_box_office.sources.audience.ingest --dry-run
.venv/bin/python -m pm_box_office.sources.audience.ingest --max-movies 10
```

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

The importer discovers recent articles from the category RSS feed and
automatically falls back to paginated archive pages when the requested start
date predates the feed window. If plain HTTP is blocked by Cloudflare, the
default `--fetch-mode auto` retries with Playwright and still writes raw
responses to `data/raw/boxofficepro`:

```sh
.venv/bin/playwright install chromium
.venv/bin/python -m pm_box_office.sources.boxofficepro.ingest --discovery auto --fetch-mode auto --start-date 2026-06-01 --end-date 2026-06-30
```

Use `--discovery rss` for feed-only recent runs, or `--discovery archive` for
explicit archive backfills.

Audience snapshots, after The Numbers has populated movies/releases/actuals:

```sh
.venv/bin/python -m pm_box_office.sources.audience.ingest --dry-run
.venv/bin/python -m pm_box_office.sources.audience.ingest --max-movies 10
```

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
.venv/bin/python -m pm_box_office.sources.amc.collect reset-collection-state --date 2026-06-30 --confirm-reset
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

The web app has two main surfaces:

- `/campaigns/{YYYY-MM-DD}` is the AMC collection control panel.
- `/sources` is the generic ingest dashboard for registered source scripts.

The AMC dashboard initializes AMC tables, syncs theatres, creates full-network
showtime inventory runs, builds/activates the default theatre sample, selects
movies for an exhibition date, and starts sampled seat collection. Seat-map
collection is sample-first: normal web/CLI runs collect one seat snapshot per
selected sampled showtime at 5 minutes before start, while full showtime
inventory remains the denominator for coverage and modeling. The core AMC
tables are:

- `amc_theatres` and `amc_theatre_sample_sets` /
  `amc_theatre_sample_members` for the theatre frame and weighted sample.
- `amc_movies` and `amc_showtimes` for showtime inventory by theatre, movie,
  exhibition date, start time, attributes, and format flags.
- `collection_campaigns`, `campaign_movies`, `collection_runs`, and
  `collection_tasks` for queueing and tracking inventory/seat collection work.
- `amc_seat_snapshots` for observed seat-map totals, unavailable/occupied
  seats, fill-rate/occupancy proxy, timing, parser metadata, and raw cache path.
- `analytics.amc_movie_day_blocks_v1` and related analytics views for same-day
  movie/day features.

Use `reset-collection-state --date YYYY-MM-DD --confirm-reset` to clear stale
campaign queue state and date-scoped seat snapshots while preserving theatres,
the fixed theatre sample, movies, and showtime inventory. Add
`--keep-seat-snapshots` when only queue state should be reset.

The `/sources` dashboard seeds source definitions from
`pm_box_office.orchestration.registry`, starts each source through
`pm_box_office.orchestration.supervisor`, and records process state/logs in:

- `ingest_sources`
- `ingest_runs`
- `ingest_run_logs`
- `source_freshness`

Registered sources are `the_numbers`, `boxofficepro`, `wikipedia`, `audience`,
and `amc_worker`. Wikipedia and audience runs require The Numbers movies first.
Freshness metrics currently read from `daily_box_office`,
`boxofficepro_weekend_predictions`, `wiki_pageviews_daily`,
`imdb_title_snapshots`, `letterboxd_film_snapshots`, and `amc_seat_snapshots`.

Local worker controls:

```sh
AMC_LOCAL_WORKER_COUNT=1 AMC_LOCAL_WORKER_MAX=2 AMC_WORKER_BATCH_LIMIT=1 AMC_WORKER_DELAY_SECONDS=3.0 \
  .venv/bin/uvicorn pm_box_office.web.app:app --reload --host 127.0.0.1 --port 8000
```

The dashboard auto-starts local worker slots when the AMC queue has due, late,
or high-overlap scheduled backlog. The defaults are conservative because missing
seat payloads are treated as backoff pressure: `AMC_LOCAL_WORKER_COUNT=1`,
`AMC_LOCAL_WORKER_MAX=2`, `AMC_WORKER_BATCH_LIMIT=1`, and
`AMC_WORKER_DELAY_SECONDS=3.0`. `AMC_AUTOSCALE_DUE_PER_WORKER` defaults to 80
due tasks per worker, `AMC_AUTOSCALE_LATE_PER_WORKER` defaults to 40 late tasks
per worker, and `AMC_AUTOSCALE_PEAK_PER_WORKER` defaults to 220 scheduled tasks
in the same minute per worker. The dashboard reads the backoff diagnostics log
and caps autoscaling when seat payload misses, failed seat tasks, HTTP retries,
or HTTP failures appear. Set `AMC_AUTOSCALE_ENABLED=false` to keep only the
baseline worker count. Each worker claims distinct queue rows from Postgres.

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
