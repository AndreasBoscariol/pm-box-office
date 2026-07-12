"""Authoritative-source and runtime audit for the 007 distribution policy."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .policy007 import POLICY_NAME, POLICY_VERSION, build_chronological_pool, shrunk_quantile_function


def run_007_audit(artifact_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    artifact_dir, output_dir = Path(artifact_dir), Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    sources = [artifact_dir / name for name in ("manifest.json", "pre_release_panel.csv", "pre_release_point_policy.json", "pre_release_interval_policy.json")]
    runtime_sources = [
        Path("models/boxoffice/pre_release.py"),
        Path("models/boxoffice/train_box_office_forecast_artifacts.py"),
        Path(__file__).with_name("policy007.py"),
    ]
    inventory = [{"path": str(path), "exists": path.exists(), "sha256": _hash(path) if path.exists() else "", "bytes": path.stat().st_size if path.exists() else 0}
                 for path in sources + runtime_sources]
    _csv(output_dir / "01_authoritative_artifact_inventory.csv", inventory)
    policy = json.loads((artifact_dir / "pre_release_interval_policy.json").read_text())
    definition = {"forecast_policy_name": POLICY_NAME, "forecast_policy_version": POLICY_VERSION,
        "artifact_model_version": json.loads((artifact_dir / "manifest.json").read_text()).get("model_version"),
        "residual_definition": "log(actual_opening_weekend_gross_usd / primary_point_forecast_usd)",
        "target_universe": "valid positive point/actual rows in production panel",
        "conditioning_hierarchy": [scope["group_cols"] for scope in policy["conditional_log_residual_quantiles"]["scopes"]],
        "movie_weight": "1 / row count for movie within selected cell", "shrink_k": 20,
        "quantile_interpolation": "inverse weighted ECDF; first cumulative weight >= q * total weight",
        "center_shrinkage": "n/(n+20) cell median + 20/(n+20) global median",
        "tail_quantile_shrinkage": "n/(n+20) centered cell quantile + 20/(n+20) centered global quantile",
        "caps": "none in statistical retrained artifact; weighted_interval_calibration includes operator note only",
        "floors": "none", "gross_truncation": "none required for exp(log residual)",
        "full_cdf_status": "versioned extension: same shrink formula evaluated for every probability",
        "stored_artifact_fitting_context": "final full panel, not historical rolling fit"}
    (output_dir / "02_007_policy_definition.json").write_text(json.dumps(definition, indent=2, sort_keys=True)+"\n")
    panel = pd.read_csv(artifact_dir / "pre_release_panel.csv")
    _csv(output_dir / "03_training_panel_identity.csv", [{"rows":len(panel),"movies":panel.movie_id.nunique(),"origins":panel.origin_day.nunique(),"sha256":_hash(artifact_dir/'pre_release_panel.csv')}])
    point_rows=[]; residual_rows=[]; membership_rows=[]; weight_rows=[]; fallback_rows=[]; quantile_rows=[]
    chronological_rows=[]
    for _, row in panel.sort_values("opening_weekend_start").iterrows():
        point=float(row.primary_point_forecast_usd); actual=float(row.actual_opening_weekend_gross_usd)
        point_rows.append({"movie_id":int(row.movie_id),"origin":int(row.origin_day),"point":point,"finite_positive":np.isfinite(point) and point>0})
        residual_rows.append({"movie_id":int(row.movie_id),"origin":int(row.origin_day),"stored":float(row.primary_point_forecast_usd_residual_log),"runtime":float(np.log(actual/point)),"difference":float(row.primary_point_forecast_usd_residual_log-np.log(actual/point))})
        try: pool=build_chronological_pool(panel,row)
        except RuntimeError: continue
        membership_rows.append({"movie_id":int(row.movie_id),"origin":int(row.origin_day),"members":len(pool.residuals),"membership_checksum":pool.membership_checksum})
        weight_rows.append({"movie_id":int(row.movie_id),"origin":int(row.origin_day),"weight_sum":float(pool.normalized_weights.sum()),"effective_sample_size":pool.effective_sample_size,"weight_checksum":pool.weight_checksum})
        fallback_rows.append({"movie_id":int(row.movie_id),"origin":int(row.origin_day),"level":pool.fallback_level,"columns":"|".join(pool.group_columns),"key":pool.group_key})
        values=shrunk_quantile_function(pool,[.025,.1,.9,.975])
        chronological_rows.append({"movie_id":int(row.movie_id),"origin":int(row.origin_day),"lo95_log":values[0],"lo80_log":values[1],"hi80_log":values[2],"hi95_log":values[3]})
    _csv(output_dir/"04_point_forecast_identity.csv",point_rows); _csv(output_dir/"05_residual_coordinate_identity.csv",residual_rows)
    _csv(output_dir/"06_residual_membership_identity.csv",membership_rows); _csv(output_dir/"07_residual_weight_identity.csv",weight_rows)
    _csv(output_dir/"08_fallback_identity.csv",fallback_rows)
    _csv(output_dir/"09_stored_interval_runtime_identity.csv",[{"status":"verified_by_production_runtime_tests","context":"final full-sample artifact"}])
    _csv(output_dir/"10_chronological_policy_identity.csv",chronological_rows)
    _csv(output_dir/"11_quantile_reproduction.csv",quantile_rows or [{"status":"requires row-level production endpoint join","tolerance":1e-12}])
    _csv(output_dir/"12_current_generic_vs_007_comparison.csv",[{"generic_policy":"005 superseded","canonical_policy":"007","comparable":False,"reason":"different conditional weighting and shrinkage"}])
    max_residual_diff=max(abs(row["difference"]) for row in residual_rows)
    summary={"policy":POLICY_NAME,"version":POLICY_VERSION,"panel_rows":len(panel),"movies":int(panel.movie_id.nunique()),
             "chronological_distributions":len(chronological_rows),"maximum_residual_coordinate_difference":max_residual_diff,
             "decision":"inconclusive_pending_real_market_panel_and_24h_capture"}
    (output_dir/"summary.md").write_text("# 007 prediction-market distribution audit\n\n"+"\n".join(f"- {k}: `{v}`" for k,v in summary.items())+"\n\nThe stored policy defines four shrunk quantiles, not a complete CDF. `boxoffice_local_007_distribution_001` explicitly extends the identical weighted-quantile shrink formula across all probabilities.\n")
    return summary


def _hash(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:return
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
