#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
exec .venv/bin/python scripts/ingest/scrape_the_numbers.py "$@"
