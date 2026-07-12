from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.boxoffice.distribution_tail_safety import TailSafeDistribution, audited_tail_log_scale
from eda.Prior.locked_distribution_panel_evaluation import Distribution, mix_distributions


LEVELS = np.r_[0.001, np.linspace(0.005, 0.995, 199), 0.999]


def example() -> TailSafeDistribution:
    point = 40_000_000.0
    base = point * np.exp(np.linspace(-1.2, 1.2, len(LEVELS)))
    base *= point / np.interp(0.5, LEVELS, base)
    return TailSafeDistribution(base, LEVELS, point, 0.45)


def test_median_and_cdf_invariants() -> None:
    dist = example()
    assert float(dist.cdf(dist.point_forecast_usd)) == pytest.approx(0.5, abs=1e-12)
    values = np.r_[-np.inf, 0.0, np.geomspace(1, 1e10, 1000), np.inf]
    cdf = dist.cdf(values)
    assert cdf[0] == 0
    assert cdf[-1] == 1
    assert np.all((cdf >= 0) & (cdf <= 1))
    assert np.all(np.diff(cdf) >= 0)


def test_quantiles_and_bucket_coherence() -> None:
    dist = example()
    quantiles = dist.ppf(LEVELS)
    assert np.all(quantiles >= 0)
    assert np.all(np.diff(quantiles) >= 0)
    edges = np.r_[0, np.arange(5_000_000, 150_000_001, 5_000_000), np.inf]
    probs = dist.bucket_probabilities(edges)
    assert np.all(probs >= 0)
    assert probs.sum() == pytest.approx(1.0, abs=1e-12)
    assert probs == pytest.approx(np.diff(np.r_[0, dist.cdf(edges[1:-1]), 1]), abs=1e-12)


def test_zero_weight_is_exact_raw_identity_and_does_not_mutate_inputs() -> None:
    dist = example()
    base_before = dist.base_quantiles.copy()
    raw = TailSafeDistribution(base_before, LEVELS, dist.point_forecast_usd, dist.tail_log_scale, contamination_weight=0)
    values = np.geomspace(1, 1e9, 100)
    assert raw.cdf(values) == pytest.approx(raw.base_cdf(values), abs=0)
    assert np.array_equal(base_before, dist.base_quantiles)


def test_audited_scale_rule() -> None:
    residuals = np.linspace(-0.8, 0.8, 101)
    assert audited_tail_log_scale(residuals) > 0.03
    assert audited_tail_log_scale(np.array([0.1])) == 0.35


def test_runtime_recreates_locked_audit_mixture() -> None:
    runtime = example()
    audited = mix_distributions(
        [
            (0.98, Distribution(runtime.base_quantiles, "base")),
            (0.02, Distribution(runtime.reference_quantiles, "t4")),
        ],
        "audit",
    )
    assert runtime.final_quantiles == pytest.approx(audited.quantiles, abs=1e-9)
    edges = np.r_[0, np.arange(2_000_000, 250_000_001, 2_000_000), np.inf]
    assert runtime.bucket_probabilities(edges) == pytest.approx(
        np.diff(np.r_[0, audited.cdf(edges[1:-1]), 1]), abs=1e-12
    )
