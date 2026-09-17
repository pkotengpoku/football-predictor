from __future__ import annotations

import argparse

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

EXCLUDED = {
    "date", "season", "home_team", "away_team", "home_goals", "away_goals",
    "total_goals", "over_2_5", "odds_over_25", "odds_under_25", "market_prob_over"
}


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    cols = [c for c in df.columns if c not in EXCLUDED]
    categorical = [c for c in cols if df[c].dtype == "object" or c == "league"]
    numeric = [c for c in cols if c not in categorical]
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


def evaluate(y: pd.Series, p: np.ndarray) -> dict[str, float]:
    pred = (p >= 0.5).astype(int)
    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else np.nan,
        "accuracy": float(accuracy_score(y, pred)),
    }


def fit_calibrator(y_val: pd.Series, p_val: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip").fit(p_val, y_val)


def betting_metrics(df: pd.DataFrame, p: np.ndarray, min_edge: float = 0.04) -> dict[str, float]:
    work = df[["over_2_5", "odds_over_25", "market_prob_over"]].copy()
    work["model_prob"] = p
    work["edge"] = work.model_prob - work.market_prob_over
    bets = work[(work.edge >= min_edge) & work.odds_over_25.notna() & (work.odds_over_25 > 1)].copy()
    if bets.empty:
        return {"bets": 0, "roi": np.nan, "max_drawdown_units": np.nan}
    bets["profit"] = np.where(bets.over_2_5.eq(1), bets.odds_over_25 - 1.0, -1.0)
    equity = bets.profit.cumsum()
    drawdown = equity - equity.cummax()
    return {
        "bets": int(len(bets)),
        "roi": float(bets.profit.mean()),
        "max_drawdown_units": float(drawdown.min()),
    }


def train_once(df: pd.DataFrame, train_seasons: list[str], val_season: str, test_season: str) -> pd.DataFrame:
    train = df[df.season.astype(str).isin(train_seasons)].copy()
    val = df[df.season.astype(str).eq(str(val_season))].copy()
    test = df[df.season.astype(str).eq(str(test_season))].copy()

    for frame_name, frame in [("train", train), ("val", val), ("test", test)]:
        filtered = frame[(frame.home_matches_seen >= 5) & (frame.away_matches_seen >= 5)].copy()
        if frame_name == "train":
            train = filtered
        elif frame_name == "val":
            val = filtered
        else:
            test = filtered

    numeric, categorical = feature_columns(df)
    features = numeric + categorical

    logistic = Pipeline([
        ("pre", make_preprocessor(numeric, categorical)),
        ("model", LogisticRegression(max_iter=2000, C=1.0)),
    ])
    logistic.fit(train[features], train.over_2_5)

    lgbm = Pipeline([
        ("pre", make_preprocessor(numeric, categorical)),
        ("model", LGBMClassifier(
            n_estimators=350,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=42,
            verbosity=-1,
        )),
    ])
    lgbm.fit(train[features], train.over_2_5)

    rows = []
    for name, model in [("logistic", logistic), ("lightgbm", lgbm)]:
        p_val_raw = model.predict_proba(val[features])[:, 1]
        calibrator = fit_calibrator(val.over_2_5, p_val_raw)
        p_test_raw = model.predict_proba(test[features])[:, 1]
        p_test = calibrator.predict(p_test_raw)
        metrics = evaluate(test.over_2_5, p_test)
        metrics.update(betting_metrics(test, p_test))
        metrics.update({"model": name, "val_season": val_season, "test_season": test_season, "n_test": len(test)})
        rows.append(metrics)

        if name == "lightgbm":
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump({
                "model": model,
                "calibrator": calibrator,
                "numeric_features": numeric,
                "categorical_features": categorical,
                "trained_through": test_season,
            }, MODEL_DIR / "over25_lightgbm.joblib")

    league_rates = train.groupby("league").over_2_5.mean().to_dict()
    global_rate = float(train.over_2_5.mean())
    p_base = test.league.map(league_rates).fillna(global_rate).to_numpy()
    metrics = evaluate(test.over_2_5, p_base)
    metrics.update(betting_metrics(test, p_base))
    metrics.update({"model": "league_baseline", "val_season": val_season, "test_season": test_season, "n_test": len(test)})
    rows.append(metrics)

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
