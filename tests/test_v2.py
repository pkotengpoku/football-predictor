import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from football_predictor.features import build_features
from football_predictor.model import betting_metrics, choose_calibration, feature_columns


def _row(date, home, away, hg, ag, over_odds=2.0, under_odds=1.8):
    return {
        "date": pd.Timestamp(date), "season": "2324", "league": "E0",
        "home_team": home, "away_team": away, "home_goals": hg, "away_goals": ag,
        "home_shots": 10, "away_shots": 8, "home_sot": 4, "away_sot": 3,
        "odds_over_25": over_odds, "odds_under_25": under_odds,
        "odds_over_25_open": over_odds, "odds_under_25_open": under_odds,
        "odds_over_25_close": 1.9, "odds_under_25_close": 1.9,
        "total_goals": hg + ag, "over_2_5": int(hg + ag >= 3),
    }


def test_v2_elo_venue_and_poisson_are_pre_match():
    df = pd.DataFrame([
        _row("2024-01-01", "A", "B", 4, 0),
        _row("2024-01-01", "C", "D", 1, 0),
        _row("2024-01-08", "A", "C", 2, 1),
        _row("2024-01-08", "B", "D", 0, 0),
        _row("2024-01-15", "A", "D", 1, 1),
    ])
    out = build_features(df)

    assert out.loc[0, "home_elo"] == 1500
    assert out.loc[1, "home_elo"] == 1500
    assert out.loc[2, "home_elo"] > 1500
    assert pd.isna(out.loc[0, "league_avg_goals"])
    assert out.loc[2, "league_avg_goals"] == 2.5
    assert out.loc[2, "home_home_gf_5"] == 4
    assert 0 <= out.loc[4, "poisson_prob_over25"] <= 1


def test_v2_allowlist_excludes_market_and_current_match_stats():
    df = pd.DataFrame({
        "league": ["E0"],
        "home_shots": [20],
        "home_shots_5": [12.0],
        "home_elo": [1510.0],
        "poisson_prob_over25": [0.58],
        "market_prob_close_over": [0.55],
    })
    numeric, categorical = feature_columns(df)
    selected = set(numeric + categorical)

    assert "home_shots" not in selected
    assert "market_prob_close_over" not in selected
    assert {"home_shots_5", "home_elo", "poisson_prob_over25", "league"} <= selected


def test_v2_calibration_selection_and_two_sided_bets():
    n = 120
    val = pd.DataFrame({"over_2_5": [0, 1] * 60})
    p = np.linspace(0.3, 0.7, n)
    kind, _, scores = choose_calibration(val, p)

    assert kind in {"raw", "sigmoid", "isotonic"}
    assert set(scores) == {"raw", "sigmoid", "isotonic"}

    df = pd.DataFrame({
        "over_2_5": [1, 0],
        "odds_over_25_open": [2.0, 2.0],
        "odds_under_25_open": [2.0, 2.0],
        "market_prob_open_over": [0.5, 0.5],
        "market_prob_close_over": [0.55, 0.45],
    })
    metrics = betting_metrics(df, np.array([0.6, 0.4]), 0.05)
    assert metrics["bets"] == 2
    assert metrics["bets_over"] == 1
    assert metrics["bets_under"] == 1
