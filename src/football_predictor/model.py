from __future__ import annotations

import argparse
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import MODEL_DIR, PROCESSED_DIR, REPORT_DIR

EPS = 1e-4
EDGE_THRESHOLDS = (0.02, 0.04, 0.06, 0.08, 0.10)
MIN_VALIDATION_BETS = 30

SAFE_NUMERIC_EXACT = {
    "home_rest_days", "away_rest_days", "home_matches_seen", "away_matches_seen",
    "league_avg_goals", "league_home_goals_avg", "league_away_goals_avg",
    "home_elo", "away_elo", "elo_diff", "elo_expected_home",
    "home_attack_rate", "home_defence_rate", "away_attack_rate", "away_defence_rate",
    "poisson_lambda_home", "poisson_lambda_away", "poisson_expected_total", "poisson_prob_over25",
}

SAFE_NUMERIC_PREFIXES = (
    "home_gf_", "home_ga_", "home_shots_", "home_sot_", "home_over25_rate_", "home_points_",
    "away_gf_", "away_ga_", "away_shots_", "away_sot_", "away_over25_rate_", "away_points_",
    "home_ewm_", "away_ewm_", "home_home_", "away_away_",
)


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [c for c in df.columns if c in SAFE_NUMERIC_EXACT or c.startswith(SAFE_NUMERIC_PREFIXES)]
    categorical = ["league"] if "league" in df.columns else []
    return numeric, categorical


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([("num", num_pipe, numeric), ("cat", cat_pipe, categorical)])


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def evaluate(y: pd.Series, p: np.ndarray) -> dict[str, float]:
    p = _clip(p)
    pred = (p >= 0.5).astype(int)
    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else np.nan,
        "accuracy": float(accuracy_score(y, pred)),
    }


@dataclass
class PlattCalibrator:
    model: LogisticRegression

    def predict(self, p: np.ndarray) -> np.ndarray:
        p = _clip(p)
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]


def fit_calibrator(kind: str, y: pd.Series, p: np.ndarray):
    p = _clip(p)
    if kind == "raw":
        return None
    if kind == "sigmoid":
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        model = LogisticRegression(C=1000.0, max_iter=1000)
        model.fit(logits, y)
        return PlattCalibrator(model)
    if kind == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(p, y)
    raise ValueError(f"Unknown calibration kind: {kind}")


def apply_calibrator(kind: str, calibrator, p: np.ndarray) -> np.ndarray:
    p = _clip(p)
    if kind == "raw":
        return p
    return _clip(calibrator.predict(p))


def choose_calibration(val: pd.DataFrame, p_val_raw: np.ndarray) -> tuple[str, object, dict[str, float]]:
    split = max(50, int(len(val) * 0.60))
    split = min(split, len(val) - 30)
    fit_y = val.over_2_5.iloc[:split]
    pick_y = val.over_2_5.iloc[split:]
    fit_p = p_val_raw[:split]
    pick_p = p_val_raw[split:]

    scores: dict[str, float] = {}
    for kind in ("raw", "sigmoid", "isotonic"):
        calibrator = fit_calibrator(kind, fit_y, fit_p)
        calibrated = apply_calibrator(kind, calibrator, pick_p)
        scores[kind] = float(log_loss(pick_y, calibrated, labels=[0, 1]))

    selected = min(scores, key=scores.get)
    final_calibrator = fit_calibrator(selected, val.over_2_5, p_val_raw)
    return selected, final_calibrator, scores


def _two_sided_bets(df: pd.DataFrame, p: np.ndarray, min_edge: float) -> pd.DataFrame:
    work = df.copy()
    work["model_prob"] = _clip(p)
    market_col = "market_prob_open_over" if "market_prob_open_over" in work.columns else "market_prob_over"
    over_odds_col = "odds_over_25_open" if "odds_over_25_open" in work.columns else "odds_over_25"
    under_odds_col = "odds_under_25_open" if "odds_under_25_open" in work.columns else "odds_under_25"

    work["market_open"] = work[market_col]
    work["edge"] = work.model_prob - work.market_open
    work["side"] = np.where(work.edge >= min_edge, "over", np.where(work.edge <= -min_edge, "under", "none"))
    work["bet_odds"] = np.where(work.side.eq("over"), work[over_odds_col], work[under_odds_col])
    bets = work[work.side.ne("none") & work.market_open.notna() & work.bet_odds.notna() & (work.bet_odds > 1)].copy()
    if bets.empty:
        return bets

    won = np.where(bets.side.eq("over"), bets.over_2_5.eq(1), bets.over_2_5.eq(0))
    bets["profit"] = np.where(won, bets.bet_odds - 1.0, -1.0)

    if "market_prob_close_over" in bets.columns:
        close = bets.market_prob_close_over
        bets["clv_prob"] = np.where(
            bets.side.eq("over"),
            close - bets.market_open,
            bets.market_open - close,
        )
    else:
        bets["clv_prob"] = np.nan
    return bets


def betting_metrics(df: pd.DataFrame, p: np.ndarray, min_edge: float = 0.04) -> dict[str, float]:
    bets = _two_sided_bets(df, p, min_edge)
    if bets.empty:
        return {
            "bets": 0, "bets_over": 0, "bets_under": 0, "roi": np.nan,
            "max_drawdown_units": np.nan, "avg_clv_prob": np.nan,
        }
    equity = bets.profit.cumsum()
    drawdown = equity - equity.cummax()
    return {
        "bets": int(len(bets)),
        "bets_over": int(bets.side.eq("over").sum()),
        "bets_under": int(bets.side.eq("under").sum()),
        "roi": float(bets.profit.mean()),
        "max_drawdown_units": float(drawdown.min()),
        "avg_clv_prob": float(bets.clv_prob.dropna().mean()) if bets.clv_prob.notna().any() else np.nan,
    }


def choose_edge_threshold(val_pick: pd.DataFrame, p_pick: np.ndarray) -> tuple[float, dict[float, float]]:
    rois: dict[float, float] = {}
    eligible: list[tuple[float, float]] = []
    for threshold in EDGE_THRESHOLDS:
        metrics = betting_metrics(val_pick, p_pick, threshold)
        roi = metrics["roi"]
        rois[threshold] = roi
        if metrics["bets"] >= MIN_VALIDATION_BETS and pd.notna(roi):
            eligible.append((threshold, roi))
    if not eligible:
        return 0.04, rois
    return max(eligible, key=lambda x: x[1])[0], rois


def _model_pipelines(numeric: list[str], categorical: list[str]):
    logistic = Pipeline([
        ("pre", make_preprocessor(numeric, categorical)),
        ("model", LogisticRegression(max_iter=2500, C=0.5)),
    ])
    lgbm = Pipeline([
        ("pre", make_preprocessor(numeric, categorical)),
        ("model", LGBMClassifier(
            n_estimators=300,
            learning_rate=0.025,
            num_leaves=20,
            min_child_samples=40,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
            verbosity=-1,
        )),
    ])
    return [("logistic", logistic), ("lightgbm", lgbm)]


def _market_row(test: pd.DataFrame, val_season: str, test_season: str) -> dict[str, float] | None:
    col = "market_prob_close_over" if "market_prob_close_over" in test.columns else "market_prob_over"
    valid = test[col].notna()
    if not valid.any():
        return None
    metrics = evaluate(test.loc[valid, "over_2_5"], test.loc[valid, col].to_numpy())
    metrics.update({
        "bets": 0, "bets_over": 0, "bets_under": 0, "roi": np.nan,
        "max_drawdown_units": np.nan, "avg_clv_prob": np.nan,
        "model": "closing_market", "calibration": "market", "val_cal_logloss": np.nan,
        "edge_threshold": np.nan, "val_season": val_season, "test_season": test_season,
        "n_test": int(valid.sum()), "market_coverage": float(valid.mean()),
    })
    return metrics


def train_once(df: pd.DataFrame, train_seasons: list[str], val_season: str, test_season: str) -> pd.DataFrame:
    train = df[df.season.astype(str).isin(train_seasons)].copy().sort_values("date")
    val = df[df.season.astype(str).eq(str(val_season))].copy().sort_values("date")
    test = df[df.season.astype(str).eq(str(test_season))].copy().sort_values("date")

    train = train[(train.home_matches_seen >= 5) & (train.away_matches_seen >= 5)].copy()
    val = val[(val.home_matches_seen >= 5) & (val.away_matches_seen >= 5)].copy()
    test = test[(test.home_matches_seen >= 5) & (test.away_matches_seen >= 5)].copy()

    numeric, categorical = feature_columns(df)
    features = numeric + categorical
    rows: list[dict] = []

    for name, model in _model_pipelines(numeric, categorical):
        model.fit(train[features], train.over_2_5)
        p_val_raw = model.predict_proba(val[features])[:, 1]

        calibration, calibrator, cal_scores = choose_calibration(val, p_val_raw)
        split = max(50, int(len(val) * 0.60))
        split = min(split, len(val) - 30)
        selection_calibrator = fit_calibrator(calibration, val.over_2_5.iloc[:split], p_val_raw[:split])
        p_val_pick = apply_calibrator(calibration, selection_calibrator, p_val_raw[split:])
        edge_threshold, _ = choose_edge_threshold(val.iloc[split:].copy(), p_val_pick)

        p_test_raw = model.predict_proba(test[features])[:, 1]
        p_test = apply_calibrator(calibration, calibrator, p_test_raw)
        metrics = evaluate(test.over_2_5, p_test)
        metrics.update(betting_metrics(test, p_test, edge_threshold))
        metrics.update({
            "model": name,
            "calibration": calibration,
            "val_cal_logloss": cal_scores[calibration],
            "edge_threshold": edge_threshold,
            "val_season": val_season,
            "test_season": test_season,
            "n_test": len(test),
            "market_coverage": float(test.get("market_prob_close_over", pd.Series(np.nan, index=test.index)).notna().mean()),
        })
        rows.append(metrics)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": model,
            "calibrator": calibrator,
            "calibration": calibration,
            "edge_threshold": edge_threshold,
            "numeric_features": numeric,
            "categorical_features": categorical,
            "trained_through": test_season,
        }, MODEL_DIR / f"over25_{name}.joblib")

    league_rates = train.groupby("league").over_2_5.mean().to_dict()
    global_rate = float(train.over_2_5.mean())
    p_base = test.league.map(league_rates).fillna(global_rate).to_numpy()
    metrics = evaluate(test.over_2_5, p_base)
    metrics.update(betting_metrics(test, p_base, 0.04))
    metrics.update({
        "model": "league_baseline", "calibration": "none", "val_cal_logloss": np.nan,
        "edge_threshold": 0.04, "val_season": val_season, "test_season": test_season,
        "n_test": len(test), "market_coverage": float(test.get("market_prob_close_over", pd.Series(np.nan, index=test.index)).notna().mean()),
    })
    rows.append(metrics)

    market = _market_row(test, val_season, test_season)
    if market is not None:
        rows.append(market)

    return pd.DataFrame(rows)


def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    seasons = sorted(df.season.astype(str).unique())
    results = []
    for test_idx in range(4, len(seasons)):
        test_season = seasons[test_idx]
        val_season = seasons[test_idx - 1]
        train_seasons = seasons[:test_idx - 1]
        print(f"train={train_seasons[0]}..{train_seasons[-1]} val={val_season} test={test_season}")
        results.append(train_once(df, train_seasons, val_season, test_season))
    return pd.concat(results, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PROCESSED_DIR / "matches_features.parquet"))
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    report = walk_forward(df)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "walk_forward_metrics.csv"
    report.to_csv(out, index=False)
    print("\nWalk-forward results")
    print(report.to_string(index=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
