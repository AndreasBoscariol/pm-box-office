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
- Social X/Nitter is an experimental proof of concept for observed public
  X/Twitter-clone mention samples; it is not treated as production search
  volume.
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

### Social X/Nitter Search-Buzz POC

Module: `pm_box_office.sources.social_x.ingest`

This experimental source tests whether Nitter-compatible public pages can
produce useful movie-title buzz samples. It stores observed public-page samples
and daily aggregates with collection status flags; do not interpret the counts
as complete X/Twitter search volume.

Collected data:

- Per-movie query variants with ambiguity notes.
- Raw observed post samples from cached `ntscraper` results.
- Daily sampled mention aggregates with cap and failure status.
- A Markdown feasibility report under `results/social_x/` by default.

Main database writes:

- `social_x_queries` stores generated title/context/hashtag queries.
- `social_x_posts_sample` stores normalized observed posts.
- `social_x_daily_counts` stores per-movie/day aggregates.
- `analytics.social_x_daily_features_v1` exposes rolling POC features.

Useful commands:

```sh
.venv/bin/python -m pm_box_office.sources.social_x.ingest --dry-run --movie-limit 5 --start-date 2026-07-01 --end-date 2026-07-01
.venv/bin/python -m pm_box_office.sources.social_x.ingest --offline --cache-dir data/raw/social_x --movie-limit 5
```

Public Nitter instances are unreliable and should be used slowly. The source is
registered as disabled in orchestration and hidden from the dashboard until a
feasibility run shows stable, non-empty, date-bucketed samples.

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

For a complete Boxoffice Pro backfill/reparse across RSS and the full forecast
archive, use:

```sh
.venv/bin/python -m pm_box_office.sources.boxofficepro.ingest --full-refresh --fetch-mode auto
```

`--full-refresh` forces `--refresh`, uses both RSS and archive discovery,
expands the date window to all dates, and raises the archive page cap so the
crawler walks until the archive runs out of pages.
After pages are cached, reparse cached article HTML concurrently with:

```sh
.venv/bin/python -m pm_box_office.sources.boxofficepro.ingest --full-refresh --offline --parse-workers 8
```

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

Build deployed opening-window forecast artifacts:

```sh
.venv/bin/python -m pm_box_office.models.opening_weekend.backtest \
  --metrics-csv results/papers/day_by_day_opening_weekend/<run>/day_by_day_metrics_by_horizon.csv
```

The deployed forecast path is `pm_box_office.models.opening_weekend`. It uses a
versioned registry, explicit 3-day/4-day/5-day target windows, BOP-size
segments, one-day-lag actual availability, and production/shadow routing. The
older standalone competition and Boxoffice Pro evaluation CLIs are deprecated as
forecast entrypoints; their useful metrics and features are folded into the
day-by-day backtest and deployed registry artifacts.

Find Polymarket movie/box-office accounts:

```sh
.venv/bin/python -m pm_box_office.sources.polymarket.accounts
```

## Research: Opening-Weekend Competition Checkpoints

The current retained competition experiment lives in:

```sh
.venv/bin/python -m pm_box_office.research.papers.recreate_day_by_day_opening_weekend \
  --snapshot-days -1,1,2 \
  --train-start-year 2022 \
  --train-end-year 2024 \
  --test-start-year 2025 \
  --test-end-year 2026 \
  --min-opening-day-gross 0 \
  --min-bop-forecast-midpoint 5000000 \
  --out results/papers/day_by_day_opening_weekend_train2022_2024_test2025_2026_bop5m
```

This run trains on 2022-2024 and tests on 2025-2026, limited to movies with a
Boxoffice Pro opening-weekend midpoint of at least $5M. The checkpoint question
is: after conditioning on the BOP estimate and the actuals known so far, does a
competition signal improve final opening-weekend gross prediction?

The retained checkpoint artifacts are written under:

- `results/papers/day_by_day_opening_weekend_train2022_2024_test2025_2026_bop5m`
- `results/papers/day_by_day_opening_weekend_train2022_2025_test2026_bop5m`

The primary files are:

- `competitive_checkpoint_headline.csv`
- `competitive_checkpoint_metrics.csv`
- `competitive_checkpoint_predictions.csv`
- `competitive_checkpoint_coefficients.csv`

Headline results for the successful 2022-2024 train / 2025-2026 test run:

| Checkpoint | Baseline | Best competition model | Best competition | MAPE lift |
| --- | ---: | --- | ---: | ---: |
| Pre-release, BOP only | 34.95% | share-attraction | 34.27% | +0.68 pts |
| Friday actual known | 16.84% | residual competition | 17.19% | -0.35 pts |
| Friday and Saturday actuals known | 9.42% | residual competition | 8.32% | +1.11 pts |

The checkpoint models predict the BOP residual:

```text
log(actual opening weekend gross) - log(Boxoffice Pro midpoint)
```

All checkpoint predictions are reconciled upward when known actual gross already
exceeds the BOP midpoint. The three competition variants are:

- `*_competition`: residualized lagged competition. It first predicts expected
  7-day competitor gross from the focal movie's BOP midpoint using train rows
  only, then uses the residual as the competition surprise signal.
- `*_share_attraction`: a Krider/Weinberg-inspired proxy. It calculates the
  competitor share of an attraction set using focal BOP midpoint, recent
  competitor grosses, the top competitor, and background competition.
- `*_logit_demand`: an Einav-inspired proxy. It calculates the log focal
  attractiveness relative to the competitive choice set:
  `log((focal BOP midpoint + 1) / (competitor + background + 1))`.

The useful signal appears at the Saturday-known checkpoint because the model is
no longer estimating the whole opening weekend from scratch. Friday and
Saturday actuals anchor the movie's own demand, and the competition term helps
calibrate the remaining Sunday expectation when the observed marketplace is
stronger or weaker than expected for a movie of that forecast size. Pre-release
share-attraction shows a small lift, which is consistent with the paper logic:
competition matters most when it changes the expected audience allocation
before own actuals are known. Friday-only competition did not improve this
split; Friday actuals already absorb much of the movie-specific demand signal,
while one extra competition term adds little and can overfit a small holdout.

## Research: Wikipedia+BOP Residual Timing

The retained Wikipedia timing experiment also runs through
`pm_box_office.research.papers.recreate_day_by_day_opening_weekend`:

```sh
.venv/bin/python -m pm_box_office.research.papers.recreate_day_by_day_opening_weekend \
  --snapshot-days -14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3 \
  --target-types 3_day \
  --train-start-year 2022 \
  --train-end-year 2024 \
  --test-start-year 2025 \
  --test-end-year 2026 \
  --min-opening-day-gross 0 \
  --min-bop-forecast-midpoint 5000000 \
  --out results/papers/day_by_day_wiki_bop_residual_train2022_2024_test2025_2026_bop5m
```

This adapts Mestyán, Yasseri, and Kertész's Wikipedia activity model to the
BOP residual setting. The original paper predicts movie revenue from
accumulated Wikipedia activity at movie-time `t`; here BOP remains the baseline
and Wikipedia predicts:

```text
log(actual 3-day opening weekend gross) - log(Boxoffice Pro midpoint)
```

The primary Wikipedia models are:

- `bop_residual_wiki_views`: uses only accumulated page views, `log1p(V)`.
- `bop_residual_wiki_full_activity`: uses accumulated `log1p(V)`, `log1p(U)`,
  `log1p(R)`, and `log1p(E)`, where `U` is unique human editors, `R` is
  collaborative rigor, and `E` is human edits.
- `bop_residual_wiki_full_activity_plus_theaters`: sensitivity only; it adds
  opening theaters and is excluded from the primary pre-release conclusion
  because actual theater count may not be known at early snapshots.

Pre-release headline results, train 2022-2024 / test 2025-2026 / BOP midpoint
at least $5M:

| Snapshot | Raw BOP MAPE | Views MAPE | Full activity MAPE | Best lift |
| ---: | ---: | ---: | ---: | ---: |
| `t=-2` | 34.95% | 34.86% | 30.29% | +4.65 pts |
| `t=-1` | 34.95% | 34.76% | 30.70% | +4.25 pts |
| `t=0` | 34.95% | 33.20% | 35.10% | +1.75 pts |

The full activity model works best immediately before release, consistent with
the paper's intuition that editor activity plus views captures committed public
attention before revenue is observed. On release day, views-only is better:
views are the cleanest mass-attention signal, while editor variables can add
noise in a small holdout.

Once Friday/Saturday actuals are known, the displayed forecast with train-only
remainder ratios ties across BOP and Wikipedia:

| Snapshot | Raw BOP MAPE | Views MAPE | Full activity MAPE |
| ---: | ---: | ---: | ---: |
| `t=1` | 12.47% | 12.47% | 12.47% |
| `t=2` | 4.14% | 4.14% | 4.14% |
| `t=3` | 0.00% | 0.00% | 0.00% |

This does not mean Wikipedia has no post-release value. It means the displayed
forecast policy is dominated by known actuals plus train-only weekend remainder
ratios. To test the more precise question, the experiment also predicts the
remaining-weekend residual directly:

```text
log(actual remaining weekend gross)
  - log(expected remaining weekend gross from train-only remainder ratio)
```

Direct remaining-weekend results:

| Snapshot | Baseline total MAPE | Views total MAPE | Full activity total MAPE |
| ---: | ---: | ---: | ---: |
| `t=1` | 12.47% | 10.56% | 11.38% |
| `t=2` | 4.14% | 3.77% | 4.05% |

On remaining gross itself:

| Snapshot | Baseline remaining MAPE | Views remaining MAPE | Full activity remaining MAPE |
| ---: | ---: | ---: | ---: |
| `t=1` | 22.23% | 18.79% | 19.54% |
| `t=2` | 17.61% | 15.98% | 15.87% |

The practical conclusion is: use full Wikipedia activity for pre-release
BOP-residual adjustment near release, especially `t=-2` and `t=-1`; use
views-only for post-Friday/post-Saturday remaining-weekend residual adjustment
when optimizing total weekend MAPE. Views appear to generalize better after
actuals arrive because Friday and Saturday already reveal much of the movie's
realized demand, leaving Wikipedia page views as a simple incremental attention
signal for multiplier shape.

### Wiki+Competition Residual Combinations

The same run now also writes a unified Wiki+competition comparison:

```text
results/papers/day_by_day_wiki_comp_residual_combo_train2022_2024_test2025_2026_bop5m/
  wiki_comp_combo_predictions.csv
  wiki_comp_combo_metrics.csv
  wiki_comp_combo_coefficients.csv
  wiki_comp_combo_headline.csv
  wiki_comp_improved_predictions.csv
  wiki_comp_improved_metrics.csv
  wiki_comp_improved_coefficients.csv
  wiki_comp_improved_headline.csv
```

This compares BOP-only, Wiki residuals, competition residuals, and combined
Wiki+competition residuals on the same train/test split and BOP-covered cohort.
For `t<=0`, each model predicts:

```text
log(actual 3-day opening weekend gross) - log(Boxoffice Pro midpoint)
```

For `t=1` and `t=2`, the model locks known actuals first, then predicts the
remaining-weekend residual around a train-only average remainder-ratio
baseline:

```text
log(actual remaining weekend gross)
  - log(expected remaining weekend gross from known actuals and train ratios)
```

The competition-only residuals use the two paper-inspired competition features:

- `bop_residual_comp_share_attraction`: competitor pressure as a share of the
  focal movie's BOP-implied size plus the competitive/background market.
- `bop_residual_comp_logit_demand`: log focal attraction versus
  competitor/background attraction.

The combination models put Wiki and competition terms into the same linear
residual regression. They are not forecast averages; the model estimates one
set of coefficients for the combined feature set at each snapshot day.

Headline total-weekend MAPE:

| Snapshot | Raw BOP | Wiki views | Wiki full | Comp share | Comp logit | Best combo | Best model |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `t=-2` | 34.95% | 34.86% | 30.29% | 33.06% | 34.18% | 33.49% | Wiki full |
| `t=-1` | 34.95% | 34.76% | 30.70% | 33.17% | 34.42% | 34.06% | Wiki full |
| `t=0` | 34.95% | 33.20% | 35.10% | 31.83% | 32.20% | 35.49% | Comp share |
| `t=1` | 12.47% | 10.56% | 11.38% | 12.69% | 14.79% | 11.59% | Wiki views |
| `t=2` | 4.14% | 3.77% | 4.05% | 3.99% | 4.26% | 4.06% | Wiki views |

Earlier snapshots from `t=-14` through `t=-3` are included in the CSV outputs,
but the residual models are not yet well identified under the current
BOP-midpoint `>= $5M` cohort because too few train rows have usable BOP
snapshots that early. Treat those early days as a data-coverage limitation for
now, not evidence that BOP-only is intrinsically best.

The combination models did not improve over the best single signal in this
run. The likely issue is that Wiki attention and competition pressure are
partly correlated and the near-release holdout is small, so adding both raw
signals to one OLS residual model increases coefficient variance.

The improved approach tests four safer combination strategies:

- **Stacked residual forecasts:** fit Wiki and competition residual models
  separately, then train a small meta-model on their predicted residuals.
- **Timing gate:** use the empirically best source by information day.
- **Residualized competition:** remove the part of competition pressure already
  explained by Wiki, then add only the leftover competition signal.
- **Ridge combinations:** keep raw Wiki+competition features, but shrink
  coefficients to reduce overfit.

Stacking was the only strategy that consistently improved the results. It uses
leave-one-out base-model predictions on train rows, so the meta-model learns
from out-of-sample-style residual forecasts rather than in-sample fitted noise.
This matters because Wiki and competition are related but not identical signals:
Wiki captures public attention, while competition captures market crowding and
relative theatrical room. Stacking lets each source make its own correction
first, then learns how much to trust each correction.

Best previous model versus best improved model:

| Snapshot | Previous best | Previous MAPE | Best improved | Improved MAPE | Lift |
| ---: | --- | ---: | --- | ---: | ---: |
| `t=-2` | Wiki full | 30.29% | Stacked views + share | 29.46% | +0.84 pts |
| `t=-1` | Wiki full | 30.70% | Stacked views + share | 29.36% | +1.33 pts |
| `t=0` | Comp share | 31.83% | Stacked views + logit | 29.26% | +2.57 pts |
| `t=1` | Wiki views | 10.56% | Stacked full + share | 10.05% | +0.52 pts |
| `t=2` | Wiki views | 3.77% | Stacked full + share | 3.51% | +0.26 pts |

The practical conclusion is that raw feature concatenation is too fragile for
this sample, but stacked residual forecasts are a useful way to combine the two
sources. The stacker works because it combines lower-dimensional model outputs,
not every correlated raw variable at once.

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
AMC_LOCAL_WORKER_COUNT=1 AMC_LOCAL_WORKER_MAX=1 AMC_WORKER_BATCH_LIMIT=1 AMC_WORKER_DELAY_SECONDS=3.0 \
  .venv/bin/uvicorn pm_box_office.web.app:app --reload --host 127.0.0.1 --port 8000
```

The dashboard auto-starts local worker slots when the AMC queue has due, late,
or high-overlap scheduled backlog. The defaults target about 20 seat tasks per
minute because missing seat payloads are treated as backoff pressure:
`AMC_LOCAL_WORKER_COUNT=1`, `AMC_LOCAL_WORKER_MAX=1`,
`AMC_WORKER_BATCH_LIMIT=1`, and `AMC_WORKER_DELAY_SECONDS=3.0`.
`AMC_AUTOSCALE_DUE_PER_WORKER` defaults to 80 due tasks per worker,
`AMC_AUTOSCALE_LATE_PER_WORKER` defaults to 40 late tasks per worker, and
`AMC_AUTOSCALE_PEAK_PER_WORKER` defaults to 220 scheduled tasks in the same
minute per worker. The dashboard reads the backoff diagnostics log and caps
autoscaling when seat payload misses, failed seat tasks, HTTP retries, or HTTP
failures appear. Set `AMC_LOCAL_WORKER_MAX` above `1` to allow autoscaling past
the 20 tasks/minute default, or set `AMC_AUTOSCALE_ENABLED=false` to keep only
the baseline worker count. Each worker claims distinct queue rows from Postgres.

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
