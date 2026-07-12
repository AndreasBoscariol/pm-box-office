"""Unified ``prediction-market`` command line interface."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Sequence

from .config import load_config


COMMANDS = (
    "discover", "sync-events", "match-movies", "export-review-queue", "import-review-decisions", "capture-books",
    "reconcile-books", "generate-distributions", "generate-probabilities", "generate-signals",
    "settle", "backtest-probabilities", "backtest-pnl", "generate-prospective-forecasts", "sync-prospective-actuals", "prospective-supervisor", "phase8-status", "phase9-status", "adjudicate-reviews", "closeout-capture", "acceptance-supervisor", "capture-health", "report",
    "build-historical-panel",
    "study-consensus-cdf",
)


def _date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected ISO date YYYY-MM-DD") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prediction-market", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        child = subparsers.add_parser(command)
        child.add_argument("--config")
        child.add_argument("--start-date", type=_date)
        child.add_argument("--end-date", type=_date)
        child.add_argument("--dry-run", action="store_true")
        child.add_argument("--seed", type=int)
        child.add_argument("--log-level", default="INFO")
        if command == "import-review-decisions":
            child.add_argument("--review-file", required=True)
            child.add_argument("--output", required=True)
        if command == "generate-prospective-forecasts":
            child.add_argument("--panel", default="models/boxoffice/boxoffice_local_007_weighted_interval_calibration/pre_release_panel.csv")
            child.add_argument("--artifact", default="models/boxoffice/pre_release_pointscale_cal_001")
            child.add_argument("--output", default="data/diagnostics/pre_release_pointscale_cal_prospective_v2")
        if command in {"sync-prospective-actuals","prospective-supervisor"}:
            child.add_argument("--panel", default="models/boxoffice/boxoffice_local_007_weighted_interval_calibration/pre_release_panel.csv")
            child.add_argument("--output", default="data/diagnostics/pre_release_pointscale_cal_prospective_v2")
        if command in {"phase8-status","phase9-status"}:
            child.add_argument("--output")
        if command == "closeout-capture":
            child.add_argument("--output", default="data/diagnostics/prediction_market_clob_capture_acceptance_v6")
        if command == "acceptance-supervisor":
            child.add_argument("--tokens", required=True)
            child.add_argument("--formal-output", default="data/diagnostics/prediction_market_clob_capture_acceptance_v6")
            child.add_argument("--forward-output", default="data/diagnostics/prediction_market_clob_forward_capture_v1")
        if command == "adjudicate-reviews":
            child.add_argument("--output", default="data/diagnostics/prediction_market_reviewed_event_universe_v6")
        if command == "build-historical-panel":
            child.add_argument("--review-dir", default="data/diagnostics/prediction_market_reviewed_event_universe_v6")
            child.add_argument("--price-output", default="data/diagnostics/prediction_market_historical_price_panel_v6")
            child.add_argument("--diagnostic-output", default="data/diagnostics/prediction_market_probability_diagnostic_pointscale_beta_v3")
            child.add_argument("--panel", default="models/boxoffice/boxoffice_local_007_weighted_interval_calibration/pre_release_panel.csv")
            child.add_argument("--no-fetch-prices", action="store_true")
            child.add_argument("--bootstrap-iterations", type=int, default=5000)
        if command == "study-consensus-cdf":
            child.add_argument("--panel", default="data/diagnostics/fallback_adjusted_daily_policy/locked_rolling_origin_distribution_quantile_panel.parquet")
            child.add_argument("--output", default="data/diagnostics/consensus_cdf_fixed_market_v1")
            child.add_argument("--oof")
            child.add_argument("--actual-grids", default="data/diagnostics/prediction_market_historical_price_panel_v6/10_complete_historical_panel.parquet")
            child.add_argument("--listing-origins", default="-14,-10,-7,-4", help="Comma-separated fixed listing origins")
        if command == "capture-books":
            child.add_argument("--tokens", help="Comma-separated string token IDs")
            child.add_argument("--duration-hours", type=float, default=24.1)
            child.add_argument("--output", default="data/diagnostics/prediction_market_clob_capture_acceptance_v5")
            child.add_argument("--diagnostic-only", action="store_true")
            child.add_argument("--controlled-faults", action="store_true")
            child.add_argument("--fault-delay-seconds", type=float, default=60.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start_date and args.end_date and args.start_date > args.end_date:
        raise SystemExit("--start-date must be on or before --end-date")
    config = load_config(args.config, {"random_seed": args.seed})
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(message)s")
    logging.getLogger(__name__).info(json.dumps({"event": "command_started", "command": args.command,
        "dry_run": args.dry_run, "start_date": str(args.start_date) if args.start_date else None,
        "end_date": str(args.end_date) if args.end_date else None, "seed": config.random_seed}, sort_keys=True))
    # Command services are deliberately explicit: dry-run validates configuration without writes.
    if args.dry_run:
        return 0
    if args.command == "import-review-decisions":
        from .review_import import export_decisions, import_review_csv
        decisions = import_review_csv(args.review_file)
        export_decisions(decisions, args.output)
        logging.getLogger(__name__).info(json.dumps({"event": "review_import_complete", "decisions": len(decisions),
            "output": str(Path(args.output).resolve())}, sort_keys=True))
        return 0
    if args.command == "generate-prospective-forecasts":
        from .prospective import generate_shadow_forecasts
        result=generate_shadow_forecasts(args.panel,args.artifact,args.output,configured_seed=config.random_seed)
        logging.getLogger(__name__).info(json.dumps(result,sort_keys=True));return 0
    if args.command == "phase8-status":
        from .phase8_status import build_phase8_status,initialize_phase8_external_outputs
        initialize_phase8_external_outputs();result=build_phase8_status()
        rendered=json.dumps(result,indent=2,sort_keys=True)
        if args.output:Path(args.output).write_text(rendered+"\n")
        print(rendered);return 0
    if args.command == "sync-prospective-actuals":
        from .operations import sync_prospective_actuals
        result=sync_prospective_actuals(args.panel,args.output);print(json.dumps(result,indent=2));return 0
    if args.command == "prospective-supervisor":
        import asyncio
        from .operations import run_shadow_supervisor
        asyncio.run(run_shadow_supervisor(panel_path=args.panel,output_dir=args.output));return 0
    if args.command == "phase9-status":
        from .phase9_initialize import initialize_phase9
        from .phase9_status import build_phase9_status
        initialize_phase9()
        result=build_phase9_status();rendered=json.dumps(result,indent=2,sort_keys=True)
        if args.output:Path(args.output).write_text(rendered+"\n")
        print(rendered);return 0
    if args.command == "capture-books":
        import asyncio
        from .capture_runtime import run_capture_acceptance
        if not args.tokens:raise SystemExit("capture-books requires --tokens")
        result=asyncio.run(run_capture_acceptance([token.strip() for token in args.tokens.split(",") if token.strip()],args.output,duration_hours=args.duration_hours,diagnostic_only=args.diagnostic_only,controlled_faults=args.controlled_faults,fault_delay_seconds=args.fault_delay_seconds))
        print(json.dumps(result,indent=2,default=str));return 0
    if args.command == "closeout-capture":
        from .acceptance_closeout import closeout_capture
        print(json.dumps(closeout_capture(args.output),indent=2));return 0
    if args.command == "acceptance-supervisor":
        import asyncio
        from .acceptance_supervisor import supervise_acceptance
        asyncio.run(supervise_acceptance([value.strip() for value in args.tokens.split(",") if value.strip()],args.formal_output,args.forward_output));return 0
    if args.command == "adjudicate-reviews":
        from .review_adjudication import adjudicate_review_queue
        print(json.dumps(adjudicate_review_queue(args.output),indent=2));return 0
    if args.command == "build-historical-panel":
        from .historical_panel import build_phase9c_historical_panel
        result=build_phase9c_historical_panel(args.review_dir,args.price_output,args.diagnostic_output,args.panel,fetch_prices=not args.no_fetch_prices,bootstrap_iterations=args.bootstrap_iterations,seed=config.random_seed)
        print(json.dumps(result,indent=2,sort_keys=True));return 0
    if args.command == "study-consensus-cdf":
        from .consensus_cdf import run_consensus_cdf_study
        origins = tuple(int(value.strip()) for value in args.listing_origins.split(",") if value.strip())
        result = run_consensus_cdf_study(args.panel, args.output, oof_path=args.oof, actual_market_path=args.actual_grids, listing_origins=origins)
        print(json.dumps(result, indent=2, sort_keys=True));return 0
    raise SystemExit(f"{args.command}: configure a database-backed service before non-dry-run execution")


if __name__ == "__main__":
    raise SystemExit(main())
