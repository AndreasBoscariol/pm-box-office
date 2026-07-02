"""Versioned deployed-model registry for opening-window forecasts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pm_box_office.models.opening_weekend.targets import (
    FIVE_DAY_TARGET,
    FOUR_DAY_TARGET,
    LARGE_BOP_SEGMENT,
    MID_BOP_SEGMENT,
    MISSING_BOP_SEGMENT,
    SMALL_BOP_SEGMENT,
    THREE_DAY_TARGET,
)


@dataclass(frozen=True)
class RegistryEntry:
    target_type: str
    bop_segment: str
    forecast_state: str
    model_family: str
    model_name: str
    model_version: str
    deployment_status: str
    reason: str
    train_policy: str = ""
    train_bop_floor_usd: int = 0
    live_bop_floor_usd: int = 0
    feature_set: tuple[str, ...] = ()
    min_holdout_n: int = 10
    allow_low_sample: bool = False


class ModelRegistry:
    def __init__(self, entries: list[RegistryEntry], *, version: str = "opening_window_registry_v1") -> None:
        self.version = version
        self.entries = entries
        self._by_key = {
            (entry.target_type, entry.bop_segment, entry.forecast_state): entry
            for entry in entries
        }

    def select(
        self,
        *,
        target_type: str,
        bop_segment: str,
        forecast_state: str,
    ) -> RegistryEntry:
        return (
            self._by_key.get((target_type, bop_segment, forecast_state))
            or self._by_key.get((target_type, bop_segment, "default"))
            or self._by_key.get((target_type, "any", forecast_state))
            or self._by_key.get((target_type, "any", "default"))
            or self._by_key[("any", "any", "default")]
        )

    def to_json_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "entries": [
                {
                    **asdict(entry),
                    "feature_set": list(entry.feature_set),
                }
                for entry in self.entries
            ],
        }

    @classmethod
    def from_json_payload(cls, payload: dict[str, Any]) -> "ModelRegistry":
        entries = [
            RegistryEntry(
                **{
                    **entry,
                    "feature_set": tuple(entry.get("feature_set", ())),
                }
            )
            for entry in payload.get("entries", [])
        ]
        return cls(entries, version=str(payload.get("version", "opening_window_registry_v1")))


def load_registry(path: Path) -> ModelRegistry:
    return ModelRegistry.from_json_payload(json.loads(path.read_text(encoding="utf-8")))


def write_registry(path: Path, registry: ModelRegistry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry.to_json_payload(), indent=2), encoding="utf-8")


def build_default_registry() -> ModelRegistry:
    entries: list[RegistryEntry] = []
    for target_type in (THREE_DAY_TARGET, FOUR_DAY_TARGET):
        entries.extend(
            [
                RegistryEntry(
                    target_type=target_type,
                    bop_segment=LARGE_BOP_SEGMENT,
                    forecast_state="pre_release",
                    model_family="segmented_residual",
                    model_name="ridge_full_plus_both_competition",
                    model_version="opening_window_v1_large_prerelease",
                    deployment_status="production",
                    reason="Large-release candidate confirmed with modest 2026 lift over raw BOP.",
                    train_policy="2022_start",
                    train_bop_floor_usd=15_000_000,
                    live_bop_floor_usd=25_000_000,
                    feature_set=("wiki_full_activity", "competition_share", "competition_logit", "target_days"),
                ),
                RegistryEntry(
                    target_type=target_type,
                    bop_segment=MID_BOP_SEGMENT,
                    forecast_state="pre_release",
                    model_family="conservative_bop_wiki",
                    model_name="shrunk_wiki_residual",
                    model_version="opening_window_v1_mid_prerelease",
                    deployment_status="cautious_production",
                    reason="$5M-$25M cohort is heterogeneous; use only conservative BOP-anchored correction.",
                    train_policy="2022_start",
                    train_bop_floor_usd=5_000_000,
                    live_bop_floor_usd=5_000_000,
                    feature_set=("wiki_views", "bop_segment", "target_days"),
                ),
                RegistryEntry(
                    target_type=target_type,
                    bop_segment="any",
                    forecast_state="official_actuals_available",
                    model_family="known_actuals_remainder",
                    model_name="train_only_remainder_ratio",
                    model_version="opening_window_v1_remainder",
                    deployment_status="production",
                    reason="Official actuals dominate; lock actuals and forecast only unresolved remainder.",
                    train_policy="2022_start",
                    train_bop_floor_usd=5_000_000,
                    live_bop_floor_usd=5_000_000,
                    feature_set=("known_actuals", "days_remaining", "target_days"),
                ),
                RegistryEntry(
                    target_type=target_type,
                    bop_segment="any",
                    forecast_state="final",
                    model_family="actuals",
                    model_name="final_actuals",
                    model_version="opening_window_v1_actuals",
                    deployment_status="production",
                    reason="Target window is complete.",
                    feature_set=("known_actuals",),
                ),
            ]
        )
    entries.extend(
        [
            RegistryEntry(
                target_type=FIVE_DAY_TARGET,
                bop_segment="any",
                forecast_state="default",
                model_family="shadow_target_history",
                model_name="shadow_insufficient_bop_5day_history",
                model_version="opening_window_v1_5day_shadow",
                deployment_status="shadow",
                reason="5-day deployment waits on Boxoffice Pro 5-day reparse/reingest.",
                feature_set=("target_days",),
            ),
            RegistryEntry(
                target_type="any",
                bop_segment=SMALL_BOP_SEGMENT,
                forecast_state="default",
                model_family="shadow_small_bop",
                model_name="shadow_bop_under_5m",
                model_version="opening_window_v1_shadow",
                deployment_status="shadow",
                reason="BOP midpoint below $5M is outside production v1 coverage.",
            ),
            RegistryEntry(
                target_type="any",
                bop_segment=MISSING_BOP_SEGMENT,
                forecast_state="default",
                model_family="shadow_missing_bop",
                model_name="shadow_missing_bop",
                model_version="opening_window_v1_shadow",
                deployment_status="shadow",
                reason="Production v1 requires a Boxoffice Pro midpoint.",
            ),
            RegistryEntry(
                target_type="any",
                bop_segment="any",
                forecast_state="same_day_proxy_available",
                model_family="future_amc_proxy",
                model_name="shadow_amc_same_day_proxy",
                model_version="opening_window_v1_amc_shadow",
                deployment_status="shadow",
                reason="AMC seat-fill is reserved for future out-of-sample validation.",
            ),
            RegistryEntry(
                target_type="any",
                bop_segment="any",
                forecast_state="default",
                model_family="shadow_unregistered_state",
                model_name="shadow_unregistered_state",
                model_version="opening_window_v1_shadow",
                deployment_status="shadow",
                reason="No production registry entry exists for this target/state/segment.",
            ),
        ]
    )
    return ModelRegistry(entries)


DEFAULT_REGISTRY = build_default_registry()
