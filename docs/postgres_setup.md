# PostgreSQL Setup

The ingest scripts are PostgreSQL-only. They read the database URL from:

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

Then edit `.env` if your local database URL differs:

```sh
DATABASE_URL=postgresql://localhost/pm_box_office
```

Run a smoke test:

```sh
scripts/run_scrape_the_numbers.sh --dry-run
```

The Wikipedia ingest uses the same database and should run after The Numbers
has populated the base `movies`, `release_runs`, and `daily_box_office` tables:

```sh
scripts/run_ingest_wikipedia_boxoffice.sh --dry-run
```

The repo uses a local `.venv` for Python dependencies because Homebrew Python is
externally managed. VS Code is configured to use `.venv/bin/python`.
