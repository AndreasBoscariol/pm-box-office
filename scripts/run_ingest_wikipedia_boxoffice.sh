#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"
exec .venv/bin/python -m pm_box_office.sources.wikipedia.ingest "$@"
