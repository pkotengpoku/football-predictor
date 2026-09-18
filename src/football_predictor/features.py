from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, RAW_DIR, load_config

OPEN_OVER_CANDIDATES = ["Avg>2.5", "B365>2.5", "P>2.5", "Max>2.5"]
OPEN_UNDER_CANDIDATES = ["Avg<2.5", "B365<2.5", "P<2.5", "Max<2.5"]
CLOSE_OVER_CANDIDATES = ["AvgC>2.5", "B365C>2.5", "PC>2.5", "MaxC>2.5"]
CLOSE_UNDER_CANDIDATES = ["AvgC<2.5", "B365C<2.5", "PC<2.5", "MaxC<2.5"]

ELO_INITIAL = 1500.0
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 60.0
EWMA_ALPHA = 0.35
POISSON_PRIOR_MATCHES = 5.0


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
        open_over = _first_existing(df, OPEN_OVER_CANDIDATES)
        open_under = _first_existing(df, OPEN_UNDER_CANDIDATES)
        close_over = _first_existing(df, CLOSE_OVER_CANDIDATES)
        close_under = _first_existing(df, CLOSE_UNDER_CANDIDATES)
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
            "odds_over_25_open": open_over,
            "odds_under_25_open": open_under,
            "odds_over_25_close": close_over,
            "odds_under_25_close": close_under,
            "odds_over_25": open_over,
            "odds_under_25": open_under,
        })
        frames.append(out)
    if not frames:
        raise RuntimeError("No raw CSVs found. Run the downloader first.")
    matches = pd.concat(frames, ignore_index=True)
    matches = matches.dropna(subset=["date", "home_goals", "away_goals"])
    matches["total_goals"] = matches.home_goals + matches.away_goals
    matches["over_2_5"] = (matches.total_goals >= 3).astype(int)
    return matches.sort_values(["date", "league", "home_team", "away_team"]).reset_index(drop=True)


def _dq() -> deque:
    return deque(maxlen=20)


@dataclass
class TeamState:
    gf: deque = field(default_factory=_dq)
    ga: deque = field(default_factory=_dq)
    shots: deque = field(default_factory=_dq)
    sot: deque = field(default_factory=_dq)
    over25: deque = field(default_factory=_dq)
    points: deque = field(default_factory=_dq)
    home_gf: deque = field(default_factory=_dq)
    home_ga: deque = field(default_factory=_dq)
    home_points: deque = field(default_factory=_dq)
    away_gf: deque = field(default_factory=_dq)
    away_ga: deque = field(default_factory=_dq)
    away_points: deque = field(default_factory=_dq)
    last_date: pd.Timestamp | None = None


def _clean(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if pd.notna(v)]


def _mean_last(values: Iterable[float], n: int) -> float:
    vals = _clean(list(values)[-n:])
    return float(np.mean(vals)) if vals else np.nan


def _ewm(values: Iterable[float], alpha: float = EWMA_ALPHA) -> float:
    vals = _clean(values)
    if not vals:
        return np.nan
    value = vals[0]
    for x in vals[1:]:
        value = alpha * x + (1.0 - alpha) * value
    return float(value)


def _shrunk_mean(values: Iterable[float], prior: float, prior_matches: float = POISSON_PRIOR_MATCHES) -> float:
    vals = _clean(values)
    if not vals:
        return float(prior)
    return float((sum(vals) + prior_matches * prior) / (len(vals) + prior_matches))


def _vig_free_probability(over_odds: float, under_odds: float) -> float:
    if pd.isna(over_odds) or pd.isna(under_odds) or over_odds <= 1 or under_odds <= 1:
        return np.nan
    p_over = 1.0 / float(over_odds)
    p_under = 1.0 / float(under_odds)
    return float(p_over / (p_over + p_under))


def _poisson_over25(total_lambda: float) -> float:
    if pd.isna(total_lambda) or total_lambda < 0:
        return np.nan
    under_or_equal_2 = np.exp(-total_lambda) * (1.0 + total_lambda + total_lambda**2 / 2.0)
    return float(np.clip(1.0 - under_or_equal_2, 0.0, 1.0))


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
    for name, values in (
        ("gf", state.gf),
        ("ga", state.ga),
        ("shots", state.shots),
        ("sot", state.sot),
        ("over25", state.over25),
        ("points", state.points),
    ):
        feats[f"{prefix}_ewm_{name}"] = _ewm(values)
    feats[f"{prefix}_matches_seen"] = len(state.gf)
    return feats


def _venue_features(state: TeamState, prefix: str, venue: str) -> dict[str, float]:
    if venue == "home":
        gf, ga, points = state.home_gf, state.home_ga, state.home_points
    else:
        gf, ga, points = state.away_gf, state.away_ga, state.away_points
    return {
        f"{prefix}_{venue}_gf_5": _mean_last(gf, 5),
        f"{prefix}_{venue}_ga_5": _mean_last(ga, 5),
        f"{prefix}_{venue}_points_5": _mean_last(points, 5),
        f"{prefix}_{venue}_gf_ewm": _ewm(gf),
        f"{prefix}_{venue}_ga_ewm": _ewm(ga),
        f"{prefix}_{venue}_points_ewm": _ewm(points),
    }


def _elo_expected(home_elo: float, away_elo: float) -> float:
    return float(1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + ELO_HOME_ADVANTAGE)) / 400.0)))


def _update_elo(home_elo: float, away_elo: float, home_goals: float, away_goals: float) -> tuple[float, float]:
    expected_home = _elo_expected(home_elo, away_elo)
    actual_home = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
    goal_diff = abs(float(home_goals) - float(away_goals))
    margin_multiplier = min(1.75, 1.0 + 0.15 * max(goal_diff - 1.0, 0.0))
    change = ELO_K * margin_multiplier * (actual_home - expected_home)
    return home_elo + change, away_elo - change


def _get_row_value(row, primary: str, fallback: str | None = None) -> float:
    if hasattr(row, primary):
        return getattr(row, primary)
    if fallback and hasattr(row, fallback):
        return getattr(row, fallback)
    return np.nan


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    team_states: dict[tuple[str, str], TeamState] = defaultdict(TeamState)
    elo: dict[tuple[str, str], float] = defaultdict(lambda: ELO_INITIAL)
    league_home_goals: dict[str, deque] = defaultdict(lambda: deque(maxlen=760))
    league_away_goals: dict[str, deque] = defaultdict(lambda: deque(maxlen=760))
    rows: list[dict] = []

    matches = matches.sort_values(["date", "league", "home_team", "away_team"]).reset_index(drop=True)

    for _, day_matches in matches.groupby("date", sort=True):
        pending_updates = []

        for row in day_matches.itertuples(index=False):
            home_state = team_states[(row.league, row.home_team)]
            away_state = team_states[(row.league, row.away_team)]

            feat = row._asdict()
            feat.update(_state_features(home_state, "home", row.date))
            feat.update(_state_features(away_state, "away", row.date))
            feat.update(_venue_features(home_state, "home", "home"))
            feat.update(_venue_features(away_state, "away", "away"))

            home_elo = elo[(row.league, row.home_team)]
            away_elo = elo[(row.league, row.away_team)]
            feat["home_elo"] = home_elo
            feat["away_elo"] = away_elo
            feat["elo_diff"] = home_elo - away_elo
            feat["elo_expected_home"] = _elo_expected(home_elo, away_elo)

            home_hist = league_home_goals[row.league]
            away_hist = league_away_goals[row.league]
            league_home_avg = float(np.mean(home_hist)) if home_hist else np.nan
            league_away_avg = float(np.mean(away_hist)) if away_hist else np.nan
            league_avg_total = (
                league_home_avg + league_away_avg
                if pd.notna(league_home_avg) and pd.notna(league_away_avg)
                else np.nan
            )
            feat["league_home_goals_avg"] = league_home_avg
            feat["league_away_goals_avg"] = league_away_avg
            feat["league_avg_goals"] = league_avg_total

            if pd.notna(league_home_avg) and pd.notna(league_away_avg) and league_home_avg > 0 and league_away_avg > 0:
                home_attack_rate = _shrunk_mean(home_state.home_gf, league_home_avg) / league_home_avg
                home_defence_rate = _shrunk_mean(home_state.home_ga, league_away_avg) / league_away_avg
                away_attack_rate = _shrunk_mean(away_state.away_gf, league_away_avg) / league_away_avg
                away_defence_rate = _shrunk_mean(away_state.away_ga, league_home_avg) / league_home_avg

                lambda_home = league_home_avg * home_attack_rate * away_defence_rate
                lambda_away = league_away_avg * away_attack_rate * home_defence_rate
                total_lambda = lambda_home + lambda_away
                feat["home_attack_rate"] = home_attack_rate
                feat["home_defence_rate"] = home_defence_rate
                feat["away_attack_rate"] = away_attack_rate
                feat["away_defence_rate"] = away_defence_rate
                feat["poisson_lambda_home"] = lambda_home
                feat["poisson_lambda_away"] = lambda_away
                feat["poisson_expected_total"] = total_lambda
                feat["poisson_prob_over25"] = _poisson_over25(total_lambda)
            else:
                feat["home_attack_rate"] = np.nan
                feat["home_defence_rate"] = np.nan
                feat["away_attack_rate"] = np.nan
                feat["away_defence_rate"] = np.nan
                feat["poisson_lambda_home"] = np.nan
                feat["poisson_lambda_away"] = np.nan
                feat["poisson_expected_total"] = np.nan
                feat["poisson_prob_over25"] = np.nan

            open_over = _get_row_value(row, "odds_over_25_open", "odds_over_25")
            open_under = _get_row_value(row, "odds_under_25_open", "odds_under_25")
            close_over = _get_row_value(row, "odds_over_25_close")
            close_under = _get_row_value(row, "odds_under_25_close")
            feat["market_prob_open_over"] = _vig_free_probability(open_over, open_under)
            feat["market_prob_close_over"] = _vig_free_probability(close_over, close_under)
            feat["market_prob_over"] = feat["market_prob_open_over"]

            rows.append(feat)
            pending_updates.append((row, home_state, away_state, home_elo, away_elo))

        for row, home_state, away_state, home_elo, away_elo in pending_updates:
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
            home_state.home_gf.append(row.home_goals)
            home_state.home_ga.append(row.away_goals)
            home_state.home_points.append(home_points)
            home_state.last_date = row.date

            away_state.gf.append(row.away_goals)
            away_state.ga.append(row.home_goals)
            away_state.shots.append(row.away_shots)
            away_state.sot.append(row.away_sot)
            away_state.over25.append(over)
            away_state.points.append(away_points)
            away_state.away_gf.append(row.away_goals)
            away_state.away_ga.append(row.home_goals)
            away_state.away_points.append(away_points)
            away_state.last_date = row.date

            league_home_goals[row.league].append(row.home_goals)
            league_away_goals[row.league].append(row.away_goals)
            new_home_elo, new_away_elo = _update_elo(home_elo, away_elo, row.home_goals, row.away_goals)
            elo[(row.league, row.home_team)] = new_home_elo
            elo[(row.league, row.away_team)] = new_away_elo

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
