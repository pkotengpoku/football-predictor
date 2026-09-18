import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from football_predictor.features import build_features


def test_no_same_match_leakage():
    df = pd.DataFrame([
        {"date": pd.Timestamp("2024-01-01"), "season": "2324", "league": "E0", "home_team": "A", "away_team": "B", "home_goals": 4, "away_goals": 0, "home_shots": 10, "away_shots": 2, "home_sot": 7, "away_sot": 1, "odds_over_25": 2.0, "odds_under_25": 1.8, "total_goals": 4, "over_2_5": 1},
        {"date": pd.Timestamp("2024-01-08"), "season": "2324", "league": "E0", "home_team": "A", "away_team": "B", "home_goals": 0, "away_goals": 0, "home_shots": 3, "away_shots": 3, "home_sot": 1, "away_sot": 1, "odds_over_25": 2.0, "odds_under_25": 1.8, "total_goals": 0, "over_2_5": 0},
    ])
    out = build_features(df)
    assert pd.isna(out.loc[0, "home_gf_5"])
    assert out.loc[1, "home_gf_5"] == 4
    assert out.loc[1, "away_ga_5"] == 4


def test_same_day_matches_do_not_share_results():
    df = pd.DataFrame([
        {"date": pd.Timestamp("2024-01-01"), "season": "2324", "league": "E0", "home_team": "A", "away_team": "B", "home_goals": 4, "away_goals": 0, "home_shots": 10, "away_shots": 2, "home_sot": 7, "away_sot": 1, "odds_over_25": 2.0, "odds_under_25": 1.8, "total_goals": 4, "over_2_5": 1},
        {"date": pd.Timestamp("2024-01-01"), "season": "2324", "league": "E0", "home_team": "C", "away_team": "D", "home_goals": 1, "away_goals": 0, "home_shots": 8, "away_shots": 5, "home_sot": 3, "away_sot": 2, "odds_over_25": 2.0, "odds_under_25": 1.8, "total_goals": 1, "over_2_5": 0},
        {"date": pd.Timestamp("2024-01-08"), "season": "2324", "league": "E0", "home_team": "A", "away_team": "C", "home_goals": 1, "away_goals": 1, "home_shots": 7, "away_shots": 7, "home_sot": 2, "away_sot": 2, "odds_over_25": 2.0, "odds_under_25": 1.8, "total_goals": 2, "over_2_5": 0},
    ])

    out = build_features(df)

    assert pd.isna(out.loc[0, "league_avg_goals"])
    assert pd.isna(out.loc[1, "league_avg_goals"])
    assert out.loc[2, "league_avg_goals"] == 2.5
