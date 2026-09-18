from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, RAW_DIR, load_config

ODDS_OVER_CANDIDATES = ["Avg>2.5", "B365>2.5", "P>2.5", "Max>2.5"]
ODDS_UNDER_CANDIDATES = ["Avg<2.5", "B365<2.5", "P<2.5", "Max<2.5"]


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="mixed", dayfirst=True, errors="coerce")


def load_raw_matches() -> pd.DataFrame:
    cfg = load_config()
    frames: list[pd.DataFrame] = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        try:
            season, league = path.stem.split("_", 1)
        except ValueError:
            continue
        if league not in cfg["leagues"]:
            continue
        df = pd.read_csv(path, encoding_errors="ignore")
        required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        if not required.issubset(df.columns):
            continue
        out = pd.DataFrame({
            "date": _parse_date(df["Date"]),
            "season": season,
            "league": league,
            "home_team": df["HomeTeam"].astype(str).str.strip(),
            "away_team": df["AwayTeam"].astype(str).str.strip(),
            "home_goals": pd.to_numeric(df["FTHG"], errors="coerce"),
            "away_goals": pd.to_numeric(df["FTAG"], errors="coerce"),
            "home_shots": pd.to_numeric(df.get("HS"), errors="coerce") if "HS" in df else np.nan,
            "away_shots": pd.to_numeric(df.get("AS"), errors="coerce") if "AS" in df else np.nan,
            "home_sot": pd.to_numeric(df.get("HST"), errors="coerce") if "HST" in df else np.nan,
            "away_sot": pd.to_numeric(df.get("AST"), errors="coerce") if "AST" in df else np.nan,
            "odds_over_25": _first_existing(df, ODDS_OVER_CANDIDATES),
            "odds_under_25": _first_existing(df, ODDS_UNDER_CANDIDATES),
        })
        frames.append(out)
    if not frames:
        raise RuntimeError("No raw CSVs found. Run the downloader first.")
    matches = pd.concat(frames, ignore_index=True)
    matches = matches.dropna(subset=["date", "home_goals", "away_goals"])
    matches["total_goals"] = matches.home_goals + matches.away_goals
    matches["over_2_5"] = (matches.total_goals >= 3).astype(int)
    return matches.sort_values(["date", "league", "home_team", "away_team"]).reset_index(drop=True)


@dataclass
class TeamState:
    gf: deque = field(default_factory=lambda: deque(maxlen=10))
    ga: deque = field(default_factory=lambda: deque(maxlen=10))
    shots: deque = field(default_factory=lambda: deque(maxlen=10))
    sot: deque = field(default_factory=lambda: deque(maxlen=10))
    over25: deque = field(default_factory=lambda: deque(maxlen=10))
    points: deque = field(default_factory=lambda: deque(maxlen=10))
    last_date: pd.Timestamp | None = None


def _mean_last(values: Iterable[float], n: int) -> float:
    vals = list(values)[-n:]
    vals = [v for v in vals if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan


def _state_features(state: TeamState, prefix: str, date: pd.Timestamp) -> dict[str, float]:
    rest_days = np.nan if state.last_date is None else max((date - state.last_date).days, 0)
    feats: dict[str, float] = {f"{prefix}_rest_days": rest_days}
    for n in (5, 10):
        feats[f"{prefix}_gf_{n}"] = _mean_last(state.gf, n)
        feats[f"{prefix}_ga_{n}"] = _mean_last(state.ga, n)
        feats[f"{prefix}_shots_{n}"] = _mean_last(state.shots, n)
        feats[f"{prefix}_sot_{n}"] = _mean_last(state.sot, n)
        feats[f"{prefix}_over25_rate_{n}"] = _mean_last(state.over25, n)
        feats[f"{prefix}_points_{n}"] = _mean_last(state.points, n)
    feats[f"{prefix}_matches_seen"] = len(state.gf)
    return feats


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    team_states: dict[tuple[str, str], TeamState] = defaultdict(TeamState)
    league_goal_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=380))
    rows: list[dict] = []

    for row in matches.itertuples(index=False):
        home_state = team_states[(row.league, row.home_team)]
        away_state = team_states[(row.league, row.away_team)]

        feat = row._asdict()
        feat.update(_state_features(home_state, "home", row.date))
        feat.update(_state_features(away_state, "away", row.date))

        hist = league_goal_history[row.league]
        league_avg_total = float(np.mean(hist)) if hist else np.nan
        feat["league_avg_goals"] = league_avg_total

        league_avg_team = league_avg_total / 2 if pd.notna(league_avg_total) and league_avg_total > 0 else np.nan
        if pd.notna(league_avg_team):
            home_attack = feat["home_gf_5"] / league_avg_team if pd.notna(feat["home_gf_5"]) else np.nan
            away_attack = feat["away_gf_5"] / league_avg_team if pd.notna(feat["away_gf_5"]) else np.nan
            home_def = feat["home_ga_5"] / league_avg_team if pd.notna(feat["home_ga_5"]) else np.nan
            away_def = feat["away_ga_5"] / league_avg_team if pd.notna(feat["away_ga_5"]) else np.nan
            exp_home = league_avg_team * home_attack * away_def if pd.notna(home_attack) and pd.notna(away_def) else np.nan
            exp_away = league_avg_team * away_attack * home_def if pd.notna(away_attack) and pd.notna(home_def) else np.nan
            feat["poisson_expected_total"] = exp_home + exp_away if pd.notna(exp_home) and pd.notna(exp_away) else np.nan
        else:
            feat["poisson_expected_total"] = np.nan

        if pd.notna(row.odds_over_25) and pd.notna(row.odds_under_25) and row.odds_over_25 > 1 and row.odds_under_25 > 1:
            p_over = 1 / row.odds_over_25
            p_under = 1 / row.odds_under_25
            feat["market_prob_over"] = p_over / (p_over + p_under)
        else:
            feat["market_prob_over"] = np.nan

        rows.append(feat)

        total = row.home_goals + row.away_goals
        over = float(total >= 3)
        home_points = 3.0 if row.home_goals > row.away_goals else 1.0 if row.home_goals == row.away_goals else 0.0
        away_points = 3.0 if row.away_goals > row.home_goals else 1.0 if row.home_goals == row.away_goals else 0.0

        home_state.gf.append(row.home_goals)
        home_state.ga.append(row.away_goals)
        home_state.shots.append(row.home_shots)
        home_state.sot.append(row.home_sot)
        home_state.over25.append(over)
        home_state.points.append(home_points)
        home_state.last_date = row.date

        away_state.gf.append(row.away_goals)
        away_state.ga.append(row.home_goals)
        away_state.shots.append(row.away_shots)
        away_state.sot.append(row.away_sot)
        away_state.over25.append(over)
        away_state.points.append(away_points)
        away_state.last_date = row.date

        hist.append(total)

    return pd.DataFrame(rows)


def build_and_save() -> Path:
    matches = load_raw_matches()
    features = build_features(matches)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / "matches_features.parquet"
    features.to_parquet(path, index=False)
    print(f"saved {len(features):,} rows -> {path}")
    return path


def main() -> None:
    build_and_save()


if __name__ == "__main__":
    main()
