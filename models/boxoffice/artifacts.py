"""Versioned model artifact loading and freezing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "models" / "boxoffice"
ACTIVE_MODEL_FILE = DEFAULT_ARTIFACT_ROOT / "ACTIVE_MODEL"


@dataclass(frozen=True)
class ModelArtifacts:
    model_version: str
    artifact_dir: Path
    manifest: dict[str, Any]
    pre_release_point_policy: dict[str, Any] = field(default_factory=dict)
    pre_release_interval_policy: dict[str, Any] = field(default_factory=dict)
    pre_release_distribution_policy: dict[str, Any] = field(default_factory=dict)
    daily_interval_policy: dict[str, Any] = field(default_factory=dict)
    amc_interval_policy: dict[str, Any] = field(default_factory=dict)
    thursday_preview_policy: dict[str, Any] = field(default_factory=dict)
    opening_thursday_actual_policy: dict[str, Any] = field(default_factory=dict)
    thursday_amc_preview_policy: dict[str, Any] = field(default_factory=dict)
    live_weekend_distribution_policy: dict[str, Any] = field(default_factory=dict)
    pre_release_panel: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_baseline: pd.DataFrame = field(default_factory=pd.DataFrame)
    live_plugin_nowcasts: pd.DataFrame = field(default_factory=pd.DataFrame)
    interval_calibration: pd.DataFrame = field(default_factory=pd.DataFrame)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def latest_model_version(root: Path = DEFAULT_ARTIFACT_ROOT) -> str:
    active_model_file = root / ACTIVE_MODEL_FILE.name
    if active_model_file.exists():
        model_version = active_model_file.read_text(encoding="utf-8").strip()
        if not model_version:
            raise ValueError(f"{active_model_file} is empty")
        if not (root / model_version / "manifest.json").exists():
            raise FileNotFoundError(f"Active model {model_version!r} has no manifest.json under {root}")
        return model_version
    candidates = [
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "manifest.json").exists()
    ] if root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No model artifacts found under {root}")
    return sorted(candidates)[-1]


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_or_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _manifest_path(artifact_dir: Path) -> Path:
    json_path = artifact_dir / "manifest.json"
    if json_path.exists():
        return json_path
    raise FileNotFoundError(f"Missing manifest.json in {artifact_dir}")


def _read_manifest(path: Path) -> dict[str, Any]:
    return _read_json(path, {})


def _artifact_path(artifact_dir: Path, manifest: dict[str, Any], *keys: str) -> Path | None:
    value: Any = manifest
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else artifact_dir / path


def load_model_artifacts(
    model_version: str = "latest",
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> ModelArtifacts:
    if model_version == "latest":
        model_version = latest_model_version(artifact_root)
    artifact_dir = artifact_root / model_version
    manifest = _read_manifest(_manifest_path(artifact_dir))

    pre_release = manifest.get("pre_release", {})
    daily_baseline = manifest.get("daily_baseline", {})
    amc = manifest.get("amc", {})

    point_policy_path = _artifact_path(artifact_dir, manifest, "pre_release", "point_policy_path")
    interval_policy_path = _artifact_path(artifact_dir, manifest, "pre_release", "interval_policy_path")
    distribution_policy_path = _artifact_path(artifact_dir, manifest, "pre_release", "distribution_policy_path")
    daily_interval_policy_path = _artifact_path(artifact_dir, manifest, "composition", "daily_interval_policy_path")
    amc_interval_policy_path = _artifact_path(artifact_dir, manifest, "amc", "interval_policy_path")
    pre_release_panel_path = _artifact_path(artifact_dir, manifest, "pre_release", "panel_path")
    daily_baseline_path = _artifact_path(artifact_dir, manifest, "daily_baseline", "baseline_path")
    live_plugin_path = _artifact_path(artifact_dir, manifest, "amc", "live_plugin_nowcasts_path")
    interval_calibration_path = _artifact_path(artifact_dir, manifest, "pre_release", "residual_scale_path")
    thursday_preview_policy_path = _artifact_path(artifact_dir, manifest, "thursday_preview", "policy_path")
    if thursday_preview_policy_path is None and (artifact_dir / "thursday_preview_policy.json").exists():
        thursday_preview_policy_path = artifact_dir / "thursday_preview_policy.json"
    opening_thursday_actual_policy_path = _artifact_path(artifact_dir, manifest, "opening_thursday_actual", "policy_path")
    if opening_thursday_actual_policy_path is None and (artifact_dir / "opening_thursday_actual_ratio_update_policy.json").exists():
        opening_thursday_actual_policy_path = artifact_dir / "opening_thursday_actual_ratio_update_policy.json"
    thursday_amc_preview_policy_path = _artifact_path(artifact_dir, manifest, "thursday_amc_preview", "policy_path")
    if thursday_amc_preview_policy_path is None and (artifact_dir / "thursday_amc_preview_policy.json").exists():
        thursday_amc_preview_policy_path = artifact_dir / "thursday_amc_preview_policy.json"
    live_distribution_policy_path = _artifact_path(artifact_dir, manifest, "live_weekend_distribution", "policy_path")
    if live_distribution_policy_path is None and (artifact_dir / "live_weekend_distribution_policy.json").exists():
        live_distribution_policy_path = artifact_dir / "live_weekend_distribution_policy.json"

    return ModelArtifacts(
        model_version=str(manifest.get("model_version") or model_version),
        artifact_dir=artifact_dir,
        manifest=manifest,
        pre_release_point_policy=_read_json(point_policy_path, pre_release.get("point_policy", {})) if point_policy_path else pre_release.get("point_policy", {}),
        pre_release_interval_policy=_read_json(interval_policy_path, pre_release.get("interval_policy", {})) if interval_policy_path else pre_release.get("interval_policy", {}),
        pre_release_distribution_policy=_read_json(distribution_policy_path, pre_release.get("distribution_policy", {})) if distribution_policy_path else pre_release.get("distribution_policy", {}),
        daily_interval_policy=_read_json(daily_interval_policy_path, {}) if daily_interval_policy_path else {},
        amc_interval_policy=_read_json(amc_interval_policy_path, {}) if amc_interval_policy_path else amc.get("interval_policy", {}),
        thursday_preview_policy=_read_json(thursday_preview_policy_path, {}) if thursday_preview_policy_path else {},
        opening_thursday_actual_policy=_read_json(opening_thursday_actual_policy_path, {}) if opening_thursday_actual_policy_path else {},
        thursday_amc_preview_policy=_read_json(thursday_amc_preview_policy_path, {}) if thursday_amc_preview_policy_path else {},
        live_weekend_distribution_policy=_read_json(live_distribution_policy_path, {}) if live_distribution_policy_path else {},
        pre_release_panel=_read_csv_or_parquet(pre_release_panel_path) if pre_release_panel_path else pd.DataFrame(),
        daily_baseline=_read_csv_or_parquet(daily_baseline_path) if daily_baseline_path else pd.DataFrame(),
        live_plugin_nowcasts=_read_csv_or_parquet(live_plugin_path) if live_plugin_path else pd.DataFrame(),
        interval_calibration=_read_csv_or_parquet(interval_calibration_path) if interval_calibration_path else pd.DataFrame(),
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
