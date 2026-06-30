#!/usr/bin/env python3
"""Train AMC same-day box-office ridge models from the local PostgreSQL DB."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.db import connect_database
except ModuleNotFoundError:  # Allow `python3 scripts/analytics/train_amc_boxoffice_models.py`.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.db import connect_database


DEFAULT_OUT_ROOT = Path("results/analytics/amc_boxoffice_ridge")
DEFAULT_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
CONTROL_COLUMNS = (
    "days_since_release",
    "opening_day_flag",
    "full_day_showtime_count",
    "movie_theatre_count",
    "premium_format_share",
)
CUTOFF_BLOCKS = {
    "3pm": (1,),
    "6pm": (1, 2),
    "9pm": (1, 2, 3),
    "midnight": (1, 2, 3, 4),
}


@dataclass(frozen=True)
class ModelResult:
    cutoff: str
    row_count: int
    alpha: float | None
    cv_r2_log_ratio: float | None
    cv_rmse_log_ratio: float
    cv_mae_log_ratio: float
    cv_mape_gross: float
    train_r2_log_ratio: float | None
    model_path: str


def cutoff_feature_columns(cutoff: str) -> list[str]:
    """Return the model feature columns for a cutoff."""

    if cutoff not in CUTOFF_BLOCKS:
        raise ValueError(f"Unknown cutoff {cutoff!r}; expected one of {sorted(CUTOFF_BLOCKS)}")
    columns: list[str] = []
    for block in CUTOFF_BLOCKS[cutoff]:
        columns.append(f"log1p_s{block}")
    for block in CUTOFF_BLOCKS[cutoff]:
        columns.append(f"o{block}")
    columns.extend(CONTROL_COLUMNS)
    columns.append("day_of_week")
    return columns


def target_log_ratio(official_daily_gross_usd: float, initial_estimate_usd: float) -> float:
    if official_daily_gross_usd <= 0:
        raise ValueError("official_daily_gross_usd must be positive")
    if initial_estimate_usd <= 0:
        raise ValueError("initial_estimate_usd must be positive")
    return math.log(official_daily_gross_usd / initial_estimate_usd)


def timestamped_output_dir(out_root: Path, *, timestamp: str | None = None) -> Path:
    stamp = timestamp or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return out_root / stamp


def require_training_dependencies() -> dict[str, Any]:
    cache_root = Path(tempfile.gettempdir()) / "pm-box-office-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    missing: list[str] = []
    modules: dict[str, Any] = {}
    for import_name, package_name in (
        ("joblib", "joblib"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("sklearn.base", "scikit-learn"),
        ("sklearn.compose", "scikit-learn"),
        ("sklearn.linear_model", "scikit-learn"),
        ("sklearn.metrics", "scikit-learn"),
        ("sklearn.model_selection", "scikit-learn"),
        ("sklearn.pipeline", "scikit-learn"),
        ("sklearn.preprocessing", "scikit-learn"),
    ):
        try:
            modules[import_name] = importlib.import_module(import_name)
        except ImportError:
            missing.append(package_name)
    if missing:
        unique = ", ".join(sorted(set(missing)))
        raise SystemExit(
            f"Missing analytics dependencies: {unique}. "
            "Install them with `python3 -m pip install -r requirements.txt`."
        )
    modules["matplotlib"].use("Agg")
    modules["matplotlib.pyplot"] = importlib.import_module("matplotlib.pyplot")
    return modules


def training_sql() -> str:
    return """
        WITH official_actuals AS (
            SELECT
                rr.movie_id,
                dbo.box_office_date::date AS exhibition_date,
                SUM(dbo.gross_usd)::double precision AS official_daily_gross_usd
            FROM daily_box_office dbo
            JOIN release_runs rr ON rr.release_run_id = dbo.release_run_id
            WHERE dbo.source = 'the_numbers'
              AND dbo.is_preview = 0
              AND dbo.gross_usd IS NOT NULL
              AND dbo.gross_usd > 0
            GROUP BY rr.movie_id, dbo.box_office_date::date
        ),
        opening_dates AS (
            SELECT
                rr.movie_id,
                MIN(dbo.box_office_date::date) AS opening_date
            FROM daily_box_office dbo
            JOIN release_runs rr ON rr.release_run_id = dbo.release_run_id
            WHERE dbo.source = 'the_numbers'
              AND dbo.is_preview = 0
              AND dbo.gross_usd IS NOT NULL
              AND dbo.gross_usd > 0
            GROUP BY rr.movie_id
        ),
        baseline_estimates AS (
            SELECT DISTINCT ON (movie_id, exhibition_date)
                movie_id,
                exhibition_date,
                estimate_usd::double precision AS initial_estimate_usd,
                recorded_at
            FROM movie_day_estimates
            WHERE is_baseline = TRUE
              AND estimate_usd > 0
            ORDER BY movie_id, exhibition_date, recorded_at DESC
        )
        SELECT
            src.movie_id,
            m.title,
            src.source_movie_id AS amc_movie_id,
            b.exhibition_date,
            est.initial_estimate_usd,
            act.official_daily_gross_usd,
            COALESCE(b.s1_occupied_proxy, 0)::double precision AS s1,
            COALESCE(b.c1_capacity, 0)::double precision AS c1,
            COALESCE(b.o1_occupancy, 0)::double precision AS o1,
            COALESCE(b.s2_occupied_proxy, 0)::double precision AS s2,
            COALESCE(b.c2_capacity, 0)::double precision AS c2,
            COALESCE(b.o2_occupancy, 0)::double precision AS o2,
            COALESCE(b.s3_occupied_proxy, 0)::double precision AS s3,
            COALESCE(b.c3_capacity, 0)::double precision AS c3,
            COALESCE(b.o3_occupancy, 0)::double precision AS o3,
            COALESCE(b.s4_occupied_proxy, 0)::double precision AS s4,
            COALESCE(b.c4_capacity, 0)::double precision AS c4,
            COALESCE(b.o4_occupancy, 0)::double precision AS o4,
            COALESCE(b.full_day_showtime_count, 0)::double precision AS full_day_showtime_count,
            COALESCE(b.movie_theatre_count, 0)::double precision AS movie_theatre_count,
            COALESCE(b.premium_format_share, 0)::double precision AS premium_format_share,
            (b.exhibition_date::date - od.opening_date) AS days_since_release,
            EXTRACT(ISODOW FROM b.exhibition_date::date)::integer AS day_of_week,
            (b.exhibition_date::date = od.opening_date) AS opening_day_flag,
            COALESCE(b.snapshot_coverage, 0)::double precision AS snapshot_coverage,
            COALESCE(b.late_snapshot_count, 0)::double precision AS late_snapshot_count,
            COALESCE(b.failed_snapshot_count, 0)::double precision AS failed_snapshot_count
        FROM analytics.amc_movie_day_blocks_v1 b
        JOIN movie_source_ids src
          ON src.source = 'amc'
         AND src.source_movie_id = b.amc_movie_id
        JOIN movies m ON m.movie_id = src.movie_id
        JOIN official_actuals act
          ON act.movie_id = src.movie_id
         AND act.exhibition_date = b.exhibition_date::date
        JOIN opening_dates od ON od.movie_id = src.movie_id
        JOIN baseline_estimates est
          ON est.movie_id = src.movie_id
         AND est.exhibition_date = b.exhibition_date::date
        WHERE act.official_daily_gross_usd > 0
          AND est.initial_estimate_usd > 0
        ORDER BY b.exhibition_date::date, m.title, src.source_movie_id
    """


def load_training_frame(conn: Any, pd: Any) -> Any:
    cursor = conn.execute(training_sql())
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def prepare_training_frame(df: Any, np: Any) -> Any:
    prepared = df.copy()
    numeric_columns = [
        "initial_estimate_usd",
        "official_daily_gross_usd",
        "s1",
        "c1",
        "o1",
        "s2",
        "c2",
        "o2",
        "s3",
        "c3",
        "o3",
        "s4",
        "c4",
        "o4",
        "full_day_showtime_count",
        "movie_theatre_count",
        "premium_format_share",
        "days_since_release",
        "snapshot_coverage",
        "late_snapshot_count",
        "failed_snapshot_count",
    ]
    for column in numeric_columns:
        prepared[column] = prepared[column].fillna(0).astype(float)
    for block in range(1, 5):
        prepared[f"log1p_s{block}"] = np.log1p(prepared[f"s{block}"].clip(lower=0))
        prepared[f"o{block}"] = prepared[f"o{block}"].fillna(0).clip(lower=0, upper=1)
    prepared["opening_day_flag"] = prepared["opening_day_flag"].fillna(False).astype(int)
    prepared["day_of_week"] = prepared["day_of_week"].fillna(0).astype(int).astype(str)
    prepared["target_log_ratio"] = prepared.apply(
        lambda row: target_log_ratio(row["official_daily_gross_usd"], row["initial_estimate_usd"]),
        axis=1,
    )
    return prepared


def filter_for_cutoff(df: Any, cutoff: str) -> Any:
    required_blocks = CUTOFF_BLOCKS[cutoff]
    filtered = df.copy()
    for block in required_blocks:
        filtered = filtered[(filtered[f"c{block}"] > 0) & (filtered[f"s{block}"] >= 0)]
    return filtered


def make_one_hot_encoder(sklearn_preprocessing: Any) -> Any:
    try:
        return sklearn_preprocessing.OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return sklearn_preprocessing.OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_pipeline(
    *,
    feature_columns: list[str],
    alphas: tuple[float, ...],
    modules: dict[str, Any],
) -> Any:
    compose = modules["sklearn.compose"]
    linear_model = modules["sklearn.linear_model"]
    pipeline = modules["sklearn.pipeline"]
    preprocessing = modules["sklearn.preprocessing"]
    numeric_features = [column for column in feature_columns if column != "day_of_week"]
    preprocessor = compose.ColumnTransformer(
        transformers=[
            ("num", preprocessing.StandardScaler(), numeric_features),
            ("dow", make_one_hot_encoder(preprocessing), ["day_of_week"]),
        ]
    )
    return pipeline.Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", linear_model.RidgeCV(alphas=list(alphas))),
        ]
    )


def train_cutoff_model(
    *,
    cutoff: str,
    df: Any,
    out_dir: Path,
    alphas: tuple[float, ...],
    folds: int,
    min_rows: int,
    modules: dict[str, Any],
) -> tuple[Any, ModelResult, Any]:
    np = modules["numpy"]
    joblib = modules["joblib"]
    base = modules["sklearn.base"]
    metrics = modules["sklearn.metrics"]
    model_selection = modules["sklearn.model_selection"]

    cutoff_df = filter_for_cutoff(df, cutoff)
    if len(cutoff_df) < min_rows:
        raise SystemExit(
            f"Cutoff {cutoff} has {len(cutoff_df)} training rows, below --min-rows={min_rows}."
        )

    feature_columns = cutoff_feature_columns(cutoff)
    x = cutoff_df[feature_columns]
    y = cutoff_df["target_log_ratio"]
    model = make_pipeline(feature_columns=feature_columns, alphas=alphas, modules=modules)

    if len(cutoff_df) >= 2:
        n_splits = min(folds, len(cutoff_df))
        cv = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_pred = model_selection.cross_val_predict(base.clone(model), x, y, cv=cv)
    else:
        cv_pred = np.repeat(float(y.iloc[0]), len(cutoff_df))

    model.fit(x, y)
    train_pred = model.predict(x)
    cv_predicted_gross = cutoff_df["initial_estimate_usd"].to_numpy() * np.exp(cv_pred)
    train_predicted_gross = cutoff_df["initial_estimate_usd"].to_numpy() * np.exp(train_pred)
    actual_gross = cutoff_df["official_daily_gross_usd"].to_numpy()
    residual = np.log(actual_gross) - np.log(cv_predicted_gross)
    ape = np.abs(cv_predicted_gross - actual_gross) / actual_gross

    predictions = cutoff_df[
        [
            "movie_id",
            "title",
            "amc_movie_id",
            "exhibition_date",
            "initial_estimate_usd",
            "official_daily_gross_usd",
        ]
    ].copy()
    predictions["cutoff"] = cutoff
    predictions["cv_predicted_log_ratio"] = cv_pred
    predictions["train_predicted_log_ratio"] = train_pred
    predictions["cv_predicted_gross_usd"] = cv_predicted_gross
    predictions["train_predicted_gross_usd"] = train_predicted_gross
    predictions["cv_residual_log_gross"] = residual
    predictions["cv_absolute_percentage_error"] = ape

    model_path = out_dir / f"model_{cutoff}.joblib"
    joblib.dump(model, model_path)

    cv_r2 = metrics.r2_score(y, cv_pred) if len(cutoff_df) >= 2 else None
    train_r2 = metrics.r2_score(y, train_pred) if len(cutoff_df) >= 2 else None
    result = ModelResult(
        cutoff=cutoff,
        row_count=len(cutoff_df),
        alpha=float(model.named_steps["model"].alpha_)
        if hasattr(model.named_steps["model"], "alpha_")
        else None,
        cv_r2_log_ratio=float(cv_r2) if cv_r2 is not None else None,
        cv_rmse_log_ratio=float(metrics.mean_squared_error(y, cv_pred) ** 0.5),
        cv_mae_log_ratio=float(metrics.mean_absolute_error(y, cv_pred)),
        cv_mape_gross=float(np.mean(ape)),
        train_r2_log_ratio=float(train_r2) if train_r2 is not None else None,
        model_path=str(model_path),
    )
    return model, result, predictions


def train_models_from_frame(
    df: Any,
    *,
    out_dir: Path,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    folds: int = 5,
    min_rows: int = 5,
    modules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    modules = modules or require_training_dependencies()
    np = modules["numpy"]
    pd = modules["pandas"]

    out_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_training_frame(df, np)
    if prepared.empty:
        raise SystemExit("No usable training rows were loaded from Postgres.")

    models: dict[str, Any] = {}
    metrics_by_cutoff: list[ModelResult] = []
    predictions_by_cutoff: list[Any] = []
    feature_columns = {
        cutoff: cutoff_feature_columns(cutoff)
        for cutoff in CUTOFF_BLOCKS
    }

    for cutoff in CUTOFF_BLOCKS:
        model, result, predictions = train_cutoff_model(
            cutoff=cutoff,
            df=prepared,
            out_dir=out_dir,
            alphas=alphas,
            folds=folds,
            min_rows=min_rows,
            modules=modules,
        )
        models[cutoff] = model
        metrics_by_cutoff.append(result)
        predictions_by_cutoff.append(predictions)

    all_predictions = pd.concat(predictions_by_cutoff, ignore_index=True)
    prepared.to_csv(out_dir / "training_rows.csv", index=False)
    all_predictions.to_csv(out_dir / "predictions.csv", index=False)

    metrics_payload = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "row_count_loaded": int(len(prepared)),
        "cutoffs": [result.__dict__ for result in metrics_by_cutoff],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    (out_dir / "feature_columns.json").write_text(
        json.dumps(feature_columns, indent=2),
        encoding="utf-8",
    )

    write_plots(
        predictions=all_predictions,
        metrics_payload=metrics_payload,
        models=models,
        feature_columns=feature_columns,
        out_dir=out_dir,
        modules=modules,
    )
    return metrics_payload


def feature_names_for_model(model: Any) -> list[str]:
    preprocessor = model.named_steps["preprocessor"]
    try:
        return [name.replace("num__", "").replace("dow__", "") for name in preprocessor.get_feature_names_out()]
    except Exception:
        return [f"feature_{index}" for index, _ in enumerate(model.named_steps["model"].coef_)]


def save_figure(fig: Any, out_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=160)
    fig.savefig(out_dir / f"{stem}.svg")


def write_plots(
    *,
    predictions: Any,
    metrics_payload: dict[str, Any],
    models: dict[str, Any],
    feature_columns: dict[str, list[str]],
    out_dir: Path,
    modules: dict[str, Any],
) -> None:
    plt = modules["matplotlib.pyplot"]
    np = modules["numpy"]

    cutoffs = list(CUTOFF_BLOCKS)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for axis, cutoff in zip(axes.ravel(), cutoffs):
        rows = predictions[predictions["cutoff"] == cutoff]
        axis.scatter(rows["official_daily_gross_usd"], rows["cv_predicted_gross_usd"], alpha=0.75)
        lower = float(min(rows["official_daily_gross_usd"].min(), rows["cv_predicted_gross_usd"].min()))
        upper = float(max(rows["official_daily_gross_usd"].max(), rows["cv_predicted_gross_usd"].max()))
        axis.plot([lower, upper], [lower, upper], color="#444444", linewidth=1)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(f"{cutoff}: actual vs predicted")
        axis.set_xlabel("Official gross")
        axis.set_ylabel("CV predicted gross")
    save_figure(fig, out_dir, "actual_vs_predicted_gross_by_cutoff")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for axis, cutoff in zip(axes.ravel(), cutoffs):
        rows = predictions[predictions["cutoff"] == cutoff]
        axis.hist(rows["cv_residual_log_gross"], bins=min(20, max(5, len(rows) // 2)), color="#4c78a8")
        axis.axvline(0, color="#444444", linewidth=1)
        axis.set_title(f"{cutoff}: log gross residuals")
        axis.set_xlabel("log(actual) - log(predicted)")
        axis.set_ylabel("movie-days")
    save_figure(fig, out_dir, "residual_distribution_by_cutoff")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    ape_values = [
        predictions[predictions["cutoff"] == cutoff]["cv_absolute_percentage_error"].to_numpy() * 100
        for cutoff in cutoffs
    ]
    try:
        axis.boxplot(ape_values, tick_labels=cutoffs, showmeans=True)
    except TypeError:
        axis.boxplot(ape_values, labels=cutoffs, showmeans=True)
    axis.set_title("Absolute percentage error by cutoff")
    axis.set_xlabel("cutoff")
    axis.set_ylabel("absolute percentage error")
    save_figure(fig, out_dir, "absolute_percentage_error_by_cutoff")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    metric_rows = metrics_payload["cutoffs"]
    x = np.arange(len(metric_rows))
    rmse = [row["cv_rmse_log_ratio"] for row in metric_rows]
    mape = [row["cv_mape_gross"] for row in metric_rows]
    axis.bar(x - 0.18, rmse, width=0.36, label="RMSE log ratio")
    axis.bar(x + 0.18, mape, width=0.36, label="MAPE gross")
    axis.set_xticks(x, [row["cutoff"] for row in metric_rows])
    axis.set_title("Cross-validated model error by cutoff")
    axis.set_xlabel("cutoff")
    axis.legend()
    save_figure(fig, out_dir, "cross_validated_metric_comparison")
    plt.close(fig)

    for cutoff, model in models.items():
        names = feature_names_for_model(model)
        coefs = model.named_steps["model"].coef_
        order = np.argsort(np.abs(coefs))[::-1]
        ordered_names = [names[index] for index in order]
        ordered_values = [float(coefs[index]) for index in order]
        fig, axis = plt.subplots(figsize=(10, max(5, len(ordered_names) * 0.35)))
        y_pos = np.arange(len(ordered_names))
        axis.barh(y_pos, ordered_values, color="#72b7b2")
        axis.set_yticks(y_pos, ordered_names)
        axis.invert_yaxis()
        axis.axvline(0, color="#444444", linewidth=1)
        axis.set_title(f"{cutoff}: ridge coefficient magnitude")
        axis.set_xlabel("standardized coefficient")
        save_figure(fig, out_dir, f"coefficient_magnitude_{cutoff}")
        plt.close(fig)


def parse_alphas(value: str) -> tuple[float, ...]:
    try:
        alphas = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated numeric alphas") from exc
    if not alphas:
        raise argparse.ArgumentTypeError("At least one alpha is required")
    return alphas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", help="PostgreSQL URL. Defaults to DATABASE_URL/POSTGRES_DSN/.env.")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help=f"Root output directory. Default: {DEFAULT_OUT_ROOT}",
    )
    parser.add_argument("--timestamp", help="Optional output run directory timestamp/name.")
    parser.add_argument("--folds", type=int, default=5, help="K-fold CV splits. Default: 5.")
    parser.add_argument("--min-rows", type=int, default=5, help="Minimum rows per cutoff. Default: 5.")
    parser.add_argument(
        "--alphas",
        type=parse_alphas,
        default=DEFAULT_ALPHAS,
        help="Comma-separated RidgeCV alpha grid.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    modules = require_training_dependencies()
    pd = modules["pandas"]
    conn = connect_database(args.database_url)
    try:
        frame = load_training_frame(conn, pd)
    finally:
        conn.close()
    out_dir = timestamped_output_dir(args.out_root, timestamp=args.timestamp)
    metrics_payload = train_models_from_frame(
        frame,
        out_dir=out_dir,
        alphas=args.alphas,
        folds=args.folds,
        min_rows=args.min_rows,
        modules=modules,
    )
    print(f"Wrote AMC box-office model run to {out_dir}")
    for result in metrics_payload["cutoffs"]:
        print(
            f"{result['cutoff']}: rows={result['row_count']} "
            f"rmse={result['cv_rmse_log_ratio']:.4f} "
            f"mape={result['cv_mape_gross']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
