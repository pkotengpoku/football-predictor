import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from football_predictor.model import feature_columns


def test_model_uses_only_pre_match_features():
    df = pd.DataFrame({
        "league": ["E0"],
        "home_shots": [15],
        "away_shots": [7],
        "home_sot": [8],
        "away_sot": [2],
        "home_goals": [3],
        "away_goals": [1],
        "total_goals": [4],
        "over_2_5": [1],
        "odds_over_25": [1.80],
        "market_prob_over": [0.55],
        "home_shots_5": [12.4],
        "away_shots_10": [9.7],
        "home_sot_5": [4.8],
        "away_sot_10": [3.2],
        "home_gf_5": [1.9],
        "away_ga_5": [1.4],
        "home_rest_days": [6],
        "away_matches_seen": [10],
        "league_avg_goals": [2.75],
        "poisson_expected_total": [2.91],\n        "poisson_prob_over25": [0.56],\n        "home_elo": [1512.0],\n        "elo_diff": [24.0],\n        "market_prob_close_over": [0.55],
    })

    numeric, categorical = feature_columns(df)
    selected = set(numeric + categorical)

    # Current-match/post-match information must never be model input.
    assert "home_shots" not in selected
    assert "away_shots" not in selected
    assert "home_sot" not in selected
    assert "away_sot" not in selected
    assert "home_goals" not in selected
    assert "away_goals" not in selected
    assert "total_goals" not in selected
    assert "over_2_5" not in selected
    assert "odds_over_25" not in selected
    assert "market_prob_over" not in selected\n    assert "market_prob_close_over" not in selected

    # Historical rolling features and pre-match context are allowed.
    assert "home_shots_5" in selected
    assert "away_shots_10" in selected
    assert "home_sot_5" in selected
    assert "away_sot_10" in selected
    assert "home_gf_5" in selected
    assert "away_ga_5" in selected
    assert "home_rest_days" in selected
    assert "away_matches_seen" in selected
    assert "league_avg_goals" in selected
    assert "poisson_expected_total" in selected\n    assert "poisson_prob_over25" in selected\n    assert "home_elo" in selected\n    assert "elo_diff" in selected
    assert categorical == ["league"]
