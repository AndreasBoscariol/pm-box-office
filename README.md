# PM Box Office

## AMC Scraper Web App

The AMC collection control panel is a FastAPI app at `web.app:app`. It uses the
same PostgreSQL database configuration as the ingest scripts:

1. `--database-url` when a script supports it
2. `DATABASE_URL`
3. `POSTGRES_DSN`
4. `.env` in the repo root

For local development, `.env` should look like:

```sh
DATABASE_URL=postgresql://localhost/pm_box_office
```

### Start Locally

From the repo root:

```sh
cd /Users/andreasboscariol/Desktop/PolyMarket/pm-box-office
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn web.app:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

The web app initializes the AMC tables automatically. From the dashboard, you
can sync theatres, create showtime collection runs, select movies, start seat
collection, and view worker logs. It also starts a local worker when collection
is kicked off from the UI.

### Start With Docker

```sh
docker compose up --build
```

Open:

```text
http://127.0.0.1:8000
```

Docker starts Postgres, the web app, and an AMC worker container together.

### Run A Worker Manually

If you want the worker running separately while using the local web app:

```sh
.venv/bin/python -m scripts.ingest.amc.jobs.worker --verbose
```
