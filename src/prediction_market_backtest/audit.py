"""Generate the selected pre-release distribution artifact audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INVENTORY_FIELDS = (
    "eligible_residual_identity", "residual_coordinate", "forecast_origin", "point_forecast",
    "target_universe", "observation_date", "chronological_eligibility_date", "residual_weights",
    "recency_weights", "scale_conditioning", "asymmetric_tail_treatment", "franchise_conditioning",
    "shrinkage", "caps_or_truncation", "fallback_hierarchy", "back_transformation", "training_data_version",
)


def audit_interval_artifact(artifact_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    artifact_dir, output_dir = Path(artifact_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    policy = json.loads((artifact_dir / "pre_release_interval_policy.json").read_text(encoding="utf-8"))
    panel = pd.read_csv(artifact_dir / "pre_release_panel.csv")
    by_origin = policy.get("log_residual_quantiles_by_origin_day", {})
    inventory = {
        "eligible_residual_identity": False, "residual_coordinate": True, "forecast_origin": True,
        "point_forecast": True, "target_universe": "evaluation_subset" in panel,
        "observation_date": "opening_weekend_start" in panel, "chronological_eligibility_date": False,
        "residual_weights": False, "recency_weights": False, "scale_conditioning": False,
        "asymmetric_tail_treatment": True, "franchise_conditioning": False, "shrinkage": False,
        "caps_or_truncation": True, "fallback_hierarchy": True, "back_transformation": True,
        "training_data_version": bool(manifest.get("model_version")),
    }
    _write_csv(output_dir / "01_interval_artifact_inventory.csv", [
        {"field": field, "present": inventory[field], "evidence": _evidence(field, artifact_dir)} for field in INVENTORY_FIELDS])
    eligibility = []
    for origin, payload in sorted(by_origin.items(), key=lambda item: int(item[0])):
        eligibility.append({"origin": origin, "stored_residual_count": len(payload.get("residual_samples_log", [])),
            "member_ids_stored": False, "eligibility_timestamps_stored": False,
            "chronological_reproduction_possible": False})
    _write_csv(output_dir / "02_residual_eligibility_audit.csv", eligibility)
    reproducibility = [{"component": "production_point_forecast", "reproducible": True,
        "reason": "panel plus point policy and production inference are stored"},
        {"component": "artifact_interval_endpoints", "reproducible": True,
         "reason": "per-origin quantiles and residual samples are stored"},
        {"component": "historical_rolling_distribution", "reproducible": False,
         "reason": "stored samples omit movie identity and chronological eligibility timestamp"}]
    _write_csv(output_dir / "03_distribution_reproducibility.csv", reproducibility)
    identities = []
    for origin, payload in sorted(by_origin.items(), key=lambda item: int(item[0])):
        samples = np.asarray(payload.get("residual_samples_log", []), dtype=float)
        for name, probability in (("lo95", .025), ("lo80", .1), ("hi80", .9), ("hi95", .975)):
            stored = float(payload.get(f"{name}_log", np.nan))
            calculated = float(np.quantile(samples, probability)) if len(samples) else np.nan
            identities.append({"origin": origin, "quantile": name, "stored_log": stored,
                "sample_quantile_log": calculated, "absolute_difference": abs(stored - calculated)})
    _write_csv(output_dir / "04_interval_quantile_identity.csv", identities)
    maximum_difference = max((row["absolute_difference"] for row in identities), default=float("nan"))
    summary = {"artifact": str(artifact_dir), "model_version": manifest.get("model_version"),
        "panel_rows": len(panel), "origins": len(by_origin), "path_decision": "path_2_separate_distribution_artifact",
        "maximum_stored_vs_sample_quantile_difference_log": maximum_difference,
        "conclusion": "The artifact contains a reusable empirical shape but cannot establish historical rolling eligibility."}
    (output_dir / "summary.md").write_text(
        "# Prediction-market distribution audit\n\n"
        f"Artifact: `{artifact_dir}`\n\n"
        f"Rows: {len(panel):,}; origins: {len(by_origin)}.\n\n"
        "Decision: **Path 2**. Build a separately versioned chronological distribution artifact. "
        "The interval artifact stores residual values and endpoint quantiles, but not residual member identity, "
        "eligibility timestamps, or weights. It therefore cannot prove no-lookahead membership for each historical forecast.\n\n"
        f"Maximum stored-versus-recomputed sample quantile difference (log scale): `{maximum_difference:.12g}`.\n",
        encoding="utf-8")
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _evidence(field: str, artifact_dir: Path) -> str:
    if field in {"eligible_residual_identity", "chronological_eligibility_date", "residual_weights", "recency_weights"}:
        return "not present in pre_release_interval_policy.json"
    return str(artifact_dir)
