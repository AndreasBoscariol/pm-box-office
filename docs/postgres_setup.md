# PostgreSQL Setup

Postgres is the system of record for The Numbers actuals, Wikipedia features,
AMC collection state, and model training datasets.

DB-backed commands read the database URL in this order:

1. `--database-url`
2. `DATABASE_URL`
3. `POSTGRES_DSN`
4. `.env` in the repo root

For a local Homebrew PostgreSQL install:

```sh
brew services start postgresql@16
createdb pm_box_office
cp .env.example .env
```

Then edit `.env` if needed:

```sh
DATABASE_URL=postgresql://localhost/pm_box_office
```

Install the project:

```sh
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

Smoke-test commands that do not mutate Postgres:

```sh
.venv/bin/python -m pm_box_office.sources.the_numbers.ingest --dry-run
.venv/bin/python -m pm_box_office.sources.wikipedia.ingest --dry-run
.venv/bin/python -m pm_box_office.sources.amc.collect --help
.venv/bin/python -m pm_box_office.sources.amc.jobs.worker --help
```

Commands that write to Postgres include:

```sh
.venv/bin/python -m pm_box_office.sources.the_numbers.ingest
.venv/bin/python -m pm_box_office.sources.wikipedia.ingest
.venv/bin/python -m pm_box_office.sources.amc.collect init-db
.venv/bin/python -m pm_box_office.sources.amc.collect ingest-theatres
.venv/bin/python -m pm_box_office.sources.amc.jobs.worker
.venv/bin/python -m pm_box_office.models.train
```

The Polymarket account scanner currently writes file outputs only:

```sh
.venv/bin/python -m pm_box_office.sources.polymarket.accounts
```

Legacy Python modules under `scripts/` and root `web.app` have been removed.
Use `pm_box_office.*` package entrypoints or the console scripts from
`pyproject.toml`.

