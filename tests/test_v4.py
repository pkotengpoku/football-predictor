import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from football_predictor.v4 import (
    OffsetLogisticCorrection,
    build_subgroup_report,
    choose_correction_scale,
)


def test_offset_model_keeps_market_logit_as_fixed_prior():
    rng = np.random.default_rng(7)
    n = 300
    x = rng.normal(size=(n, 1))
    market_prob = np.linspace(0.30, 0.70, n)
    market_logit = np.log(market_prob / (1.0 - market_prob))
    true_logit = market_logit + 0.8 * x[:, 0]
    y = rng.binomial(1, expit(true_logit))

    model = OffsetLogisticCorrection(C=1.0, max_iter=500)
    model.fit(x, y, market_logit)

    p_zero = model.predict_proba(x, market_logit, scale=0.0)
    assert np.allclose(p_zero, market_prob)


def test_validation_can_turn_correction_completely_off():
    y = pd.Series([0, 1] * 50)
    market = np.array([0.20, 0.80] * 50)
    market_logit = np.log(market / (1.0 - market))
    harmful_delta = np.array([2.0, -2.0] * 50)

    scale, scores = choose_correction_scale(y, market_logit, harmful_delta)

    assert scale == 0.0
    assert scores[0.0] < scores[1.0]


def test_subgroup_report_contains_requested_diagnostics():
    frame = pd.DataFrame({
        "league": ["E0", "E0", "I1", "I1"],
        "actual_over25": [1, 0, 1, 0],
        "residual_probability": [0.60, 0.40, 0.70, 0.30],
        "opening_market": [0.55, 0.45, 0.60, 0.40],
        "closing_market": [0.58, 0.42, 0.64, 0.37],
        "edge_open": [0.05, -0.05, 0.10, -0.10],
        "abs_edge_open": [0.05, 0.05, 0.10, 0.10],
        "closing_move_over": [0.03, -0.03, 0.04, -0.03],
        "bet_side": ["over", "under", "over", "under"],
        "profit": [1.0, 1.0, 1.0, 1.0],
        "clv_prob": [0.03, 0.03, 0.04, 0.03],
    })

    report = build_subgroup_report(frame)

    assert ((report.group_type == "league") & (report.group_value == "E0")).any()
    assert ((report.group_type == "bet_side") & (report.group_value == "over")).any()
    assert (report.group_type == "abs_edge").any()
