from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.boxoffice.thursday_amc_preview import predict_preview_distribution, update_ow_distribution


def test_residual_pool_fallback_and_ow_uncertainty_propagation() -> None:
    policy = {
        "min_residual_pool_n": 3,
        "residual_pools": {
            "origin:16:00": [0.1],
            "origin_bucket:late": [-0.2, 0.0, 0.2],
            "pooled_thursday": [-0.4, 0.0, 0.4],
        },
    }
    preview = predict_preview_distribution(
        point_preview_gross_usd=10_000_000,
        forecast_origin="16:00",
        policy=policy,
        draws=500,
        seed=4,
    )
    assert preview["residual_pool_scope"] == "origin_bucket:late"
    assert np.all(np.asarray(preview["draws_usd"]) > 0)
    ow = update_ow_distribution(
        baseline_ow_usd=80_000_000,
        preview_distribution=preview,
        preview_update_policy={"alpha": 0.0, "beta": 0.5},
    )
    assert ow["enabled"] is True
    assert ow["hi95_usd"] > ow["lo95_usd"]
