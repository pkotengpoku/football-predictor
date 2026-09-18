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
BLEND_WEIGHTS = tuple(np.round(np.linspace(0.0, 1.0, 21), 2))
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


def feature_columns(df: pd.DataFrame, include_market: bool = False) -> tuple[list[str], list[str]]:
    numeric = [
        c for c in df.columns
        if c in SAFE_NUMERIC_EXACT or c.startswith(SAFE_NUMERIC_PREFIXES)
    ]
    if include_market and "market_logit_open" in df.columns:
        numeric.append("market_logit_open")
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


def _clip(p: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def _logit(p: np.ndarray | pd.Series) -> np.ndarray:
    p = _clip(p)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def with_market_logit(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "market_prob_open_over" in out.columns:
        market = pd.to_numeric(out["market_prob_open_over"], errors="coerce")
    elif "market_prob_over" in out.columns:
        market = pd.to_numeric(out["market_prob_over"], errors="coerce")
    else:
        market = pd.Series(np.nan, index=out.index, dtype=float)
    valid = market.notna()
    out["market_logit_open"] = np.nan
    out.loc[valid, "market_logit_open"] = _logit(market.loc[valid])
    return out


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
        logits = _logit(p).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]


def fit_calibrator(kind: str, y: pd.Series, p: np.ndarray):
    p = _clip(p)
    if kind == "raw":
        return None
    if kind == "sigmoid":
        logits = _logit(p).reshape(-1, 1)
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


def _validation_split(val: pd.DataFrame) -> int:
    split = max(50, int(len(val) * 0.60))
    return min(split, len(val) - 30)


def choose_calibration(val: pd.DataFrame, p_val_raw: np.ndarray) -> tuple[str, object, dict[str, float]]:
    split = _validation_split(val)
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


def blend_probabilities(model_p: np.ndarray, market_p: np.ndarray, model_weight: float) -> np.ndarray:
    model_logit = _logit(model_p)
    market_logit = _logit(market_p)
    return _clip(_sigmoid(model_weight * model_logit + (1.0 - model_weight) * market_logit))


def choose_blend_weight(y: pd.Series, model_p: np.ndarray, market_p: np.ndarray) -> tuple[float, dict[float, float]]:
    y_arr = np.asarray(y)
    model_arr = np.asarray(model_p, dtype=float)
    market_arr = np.asarray(market_p, dtype=float)
    valid = np.isfinite(model_arr) & np.isfinite(market_arr)
    if valid.sum() < 30:
        return 0.0, {}
    scores: dict[float, float] = {}
    for weight in BLEND_WEIGHTS:
        p = blend_probabilities(model_arr[valid], market_arr[valid], weight)
        scores[weight] = float(log_loss(y_arr[valid], p, labels=[0, 1]))
    selected = min(scores, key=scores.get)
    return float(selected), scores


def _two_sided_bets(df: pd.DataFrame, p: np.ndarray, min_edge: float | None) -> pd.DataFrame:
    work = df.copy()
    work["model_prob"] = _clip(p)
    market_col = "market_prob_open_over" if "market_prob_open_over" in work.columns else "market_prob_over"
    over_odds_col = "odds_over_25_open" if "odds_over_25_open" in work.columns else "odds_over_25"
    under_odds_col = "odds_under_25_open" if "odds_under_25_open" in work.columns else "odds_under_25"

    work["market_open"] = work[market_col]
    work["edge"] = work.model_prob - work.market_open
    if min_edge is None or pd.isna(min_edge):
        return work.iloc[0:0].copy()

    work["side"] = np.where(
        work.edge >= min_edge,
        "over",
        np.where(work.edge <= -min_edge, "under", "none"),
    )
    work["bet_odds"] = np.where(work.side.eq("over"), work[over_odds_col], work[under_odds_col])
    bets = work[
        work.side.ne("none")
        & work.market_open.notna()
        & work.bet_odds.notna()
        & (work.bet_odds > 1)
    ].copy()
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


def betting_metrics(df: pd.DataFrame, p: np.ndarray, min_edge: float | None = 0.04) -> dict[str, float]:
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


def choose_edge_threshold(val_pick: pd.DataFrame, p_pick: np.ndarray) -> tuple[float | None, dict[float, dict[str, float]]]:
    diagnostics: dict[float, dict[str, float]] = {}
    candidates: list[tuple[float, float, float, int]] = []

    for threshold in EDGE_THRESHOLDS:
        metrics = betting_metrics(val_pick, p_pick, threshold)
        diagnostics[threshold] = metrics
        clv = metrics["avg_clv_prob"]
        roi = metrics["roi"]
        bets = metrics["bets"]
        if bets >= MIN_VALIDATION_BETS and pd.notna(clv):
            candidates.append((threshold, float(clv), float(roi) if pd.notna(roi) else -np.inf, bets))

    positive_clv = [x for x in candidates if x[1] > 0.0]
    if not positive_clv:
        return None, diagnostics

    selected = max(positive_clv, key=lambda x: (x[1], x[2], x[3]))
    return selected[0], diagnostics


def _football_pipelines(numeric: list[str], categorical: list[str]):
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


def _anchored_logistic(numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline([
        ("pre", make_preprocessor(numeric, categorical)),
        ("model", LogisticRegression(max_iter=2500, C=0.5)),
    ])


def _benchmark_row(
    test: pd.DataFrame,
    p: np.ndarray,
    model_name: str,
    val_season: str,
    test_season: str,
    coverage: float = 1.0,
) -> dict[str, float]:
    metrics = evaluate(test.over_2_5, p)
    metrics.update({
        "bets": 0, "bets_over": 0, "bets_under": 0, "roi": np.nan,
        "max_drawdown_units": np.nan, "avg_clv_prob": np.nan,
        "model": model_name, "calibration": "benchmark", "val_cal_logloss": np.nan,
        "edge_threshold": np.nan, "blend_weight": np.nan,
        "val_season": val_season, "test_season": test_season,
        "n_test": len(test), "market_coverage": coverage,
    })
    return metrics


def _market_benchmark_row(test: pd.DataFrame, col: str, name: str, val_season: str, test_season: str):
    if col not in test.columns:
        return None
    valid = test[col].notna()
    if not valid.any():
        return None
    subset = test.loc[valid].copy()
    return _benchmark_row(
        subset,
        subset[col].to_numpy(),
        name,
        val_season,
        test_season,
        coverage=float(valid.mean()),
    )


def _fit_calibrate_and_select(
    name: str,
    model,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    numeric: list[str],
    categorical: list[str],
) -> tuple[dict[str, float], dict]:
    model.fit(train[features], train.over_2_5)
    p_val_raw = model.predict_proba(val[features])[:, 1]

    calibration, calibrator, cal_scores = choose_calibration(val, p_val_raw)
    split = _validation_split(val)
    selection_calibrator = fit_calibrator(
        calibration,
        val.over_2_5.iloc[:split],
        p_val_raw[:split],
    )
    p_val_pick = apply_calibrator(
        calibration,
        selection_calibrator,
        p_val_raw[split:],
    )
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
        "blend_weight": np.nan,
        "val_season": None,
        "test_season": None,
        "n_test": len(test),
        "market_coverage": float(test.get(
            "market_prob_close_over",
            pd.Series(np.nan, index=test.index),
        ).notna().mean()),
    })
    artifact = {
        "model": model,
        "calibrator": calibrator,
        "calibration": calibration,
        "edge_threshold": edge_threshold,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "p_val_raw": p_val_raw,
        "p_test": p_test,
        "val_split": split,
    }
    return metrics, artifact


def train_once(df: pd.DataFrame, train_seasons: list[str], val_season: str, test_season: str) -> pd.DataFrame:
    train = with_market_logit(df[df.season.astype(str).isin(train_seasons)].copy().sort_values("date"))
    val = with_market_logit(df[df.season.astype(str).eq(str(val_season))].copy().sort_values("date"))
    test = with_market_logit(df[df.season.astype(str).eq(str(test_season))].copy().sort_values("date"))

    train = train[(train.home_matches_seen >= 5) & (train.away_matches_seen >= 5)].copy()
    val = val[(val.home_matches_seen >= 5) & (val.away_matches_seen >= 5)].copy()
    test = test[(test.home_matches_seen >= 5) & (test.away_matches_seen >= 5)].copy()

    rows: list[dict] = []
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    football_numeric, football_categorical = feature_columns(df, include_market=False)
    football_features = football_numeric + football_categorical
    football_artifacts: dict[str, dict] = {}

    for name, model in _football_pipelines(football_numeric, football_categorical):
        metrics, artifact = _fit_calibrate_and_select(
            name, model, train, val, test,
            football_features, football_numeric, football_categorical,
        )
        metrics["val_season"] = val_season
        metrics["test_season"] = test_season
        rows.append(metrics)
        football_artifacts[name] = artifact

        joblib.dump({
            "model": artifact["model"],
            "calibrator": artifact["calibrator"],
            "calibration": artifact["calibration"],
            "edge_threshold": artifact["edge_threshold"],
            "numeric_features": football_numeric,
            "categorical_features": football_categorical,
            "trained_through": test_season,
        }, MODEL_DIR / f"over25_{name}.joblib")

    anchored_train = train[train.market_logit_open.notna()].copy()
    anchored_val = val[val.market_logit_open.notna()].copy()
    anchored_test = test[test.market_logit_open.notna()].copy()
    anchored_numeric, anchored_categorical = feature_columns(anchored_train, include_market=True)
    anchored_features = anchored_numeric + anchored_categorical

    if len(anchored_train) and len(anchored_val) >= 80 and len(anchored_test):
        anchored_model = _anchored_logistic(anchored_numeric, anchored_categorical)
        metrics, artifact = _fit_calibrate_and_select(
            "market_anchored_logistic",
            anchored_model,
            anchored_train,
            anchored_val,
            anchored_test,
            anchored_features,
            anchored_numeric,
            anchored_categorical,
        )
        metrics["val_season"] = val_season
        metrics["test_season"] = test_season
        rows.append(metrics)

        joblib.dump({
            "model": artifact["model"],
            "calibrator": artifact["calibrator"],
            "calibration": artifact["calibration"],
            "edge_threshold": artifact["edge_threshold"],
            "numeric_features": anchored_numeric,
            "categorical_features": anchored_categorical,
            "trained_through": test_season,
        }, MODEL_DIR / "over25_market_anchored_logistic.joblib")

    log_art = football_artifacts["logistic"]
    split = log_art["val_split"]
    selection_calibrator = fit_calibrator(
        log_art["calibration"],
        val.over_2_5.iloc[:split],
        log_art["p_val_raw"][:split],
    )
    p_val_pick_football = apply_calibrator(
        log_art["calibration"],
        selection_calibrator,
        log_art["p_val_raw"][split:],
    )
    val_pick = val.iloc[split:].copy()
    market_pick = val_pick.market_prob_open_over.to_numpy()
    blend_weight, _ = choose_blend_weight(
        val_pick.over_2_5,
        p_val_pick_football,
        market_pick,
    )

    valid_pick = val_pick.market_prob_open_over.notna().to_numpy()
    p_blend_pick = np.full(len(val_pick), np.nan)
    if valid_pick.any():
        p_blend_pick[valid_pick] = blend_probabilities(
            p_val_pick_football[valid_pick],
            market_pick[valid_pick],
            blend_weight,
        )
    edge_threshold, _ = choose_edge_threshold(
        val_pick.loc[valid_pick].copy(),
        p_blend_pick[valid_pick],
    )

    valid_test = test.market_prob_open_over.notna().to_numpy()
    if valid_test.any():
        blend_test = test.loc[valid_test].copy()
        p_blend_test = blend_probabilities(
            log_art["p_test"][valid_test],
            blend_test.market_prob_open_over.to_numpy(),
            blend_weight,
        )
        metrics = evaluate(blend_test.over_2_5, p_blend_test)
        metrics.update(betting_metrics(blend_test, p_blend_test, edge_threshold))
        metrics.update({
            "model": "market_blend_logistic",
            "calibration": log_art["calibration"],
            "val_cal_logloss": np.nan,
            "edge_threshold": edge_threshold,
            "blend_weight": blend_weight,
            "val_season": val_season,
            "test_season": test_season,
            "n_test": len(blend_test),
            "market_coverage": float(valid_test.mean()),
        })
        rows.append(metrics)

    league_rates = train.groupby("league").over_2_5.mean().to_dict()
    global_rate = float(train.over_2_5.mean())
    p_base = test.league.map(league_rates).fillna(global_rate).to_numpy()
    rows.append(_benchmark_row(
        test, p_base, "league_baseline", val_season, test_season,
        coverage=float(test.market_prob_open_over.notna().mean()),
    ))

    if "poisson_prob_over25" in test.columns:
        valid_poisson = test.poisson_prob_over25.notna()
        if valid_poisson.any():
            subset = test.loc[valid_poisson].copy()
            rows.append(_benchmark_row(
                subset,
                subset.poisson_prob_over25.to_numpy(),
                "poisson_baseline",
                val_season,
                test_season,
                coverage=float(valid_poisson.mean()),
            ))

    opening = _market_benchmark_row(
        test, "market_prob_open_over", "opening_market", val_season, test_season
    )
    if opening is not None:
        rows.append(opening)

    closing = _market_benchmark_row(
        test, "market_prob_close_over", "closing_market", val_season, test_season
    )
    if closing is not None:
        rows.append(closing)

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
