import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from football_predictor.model import (
    blend_probabilities,
    choose_blend_weight,
    choose_edge_threshold,
    feature_columns,
    with_market_logit,
)


def test_market_feature_only_enters_anchored_models():
    df = pd.DataFrame({
        "league": ["E0"],
        "home_elo": [1510.0],
        "market_prob_open_over": [0.60],
    })
    enriched = with_market_logit(df)

    numeric_plain, _ = feature_columns(enriched, include_market=False)
    numeric_market, _ = feature_columns(enriched, include_market=True)

    assert "market_logit_open" not in numeric_plain
    assert "market_logit_open" in numeric_market


def test_blend_weight_can_prefer_market():
    y = pd.Series([0, 1] * 50)
    market = np.array([0.2, 0.8] * 50)
    model = np.array([0.8, 0.2] * 50)

    weight, scores = choose_blend_weight(y, model, market)

    assert weight == 0.0
    assert scores[0.0] < scores[1.0]


def test_logit_blend_stays_between_zero_and_one():
    p = blend_probabilities(
        np.array([0.25, 0.75]),
        np.array([0.50, 0.50]),
        0.5,
    )
    assert np.all((p > 0) & (p < 1))


def test_threshold_can_choose_no_bet_when_clv_is_not_positive():
    n = 100
    val = pd.DataFrame({
        "over_2_5": [0, 1] * 50,
        "odds_over_25_open": [2.0] * n,
        "odds_under_25_open": [2.0] * n,
        "market_prob_open_over": [0.5] * n,
        "market_prob_close_over": [0.5] * n,
    })
    p = np.array([0.60, 0.40] * 50)

    threshold, diagnostics = choose_edge_threshold(val, p)

    assert threshold is None
    assert 0.02 in diagnostics
