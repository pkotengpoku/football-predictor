from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy import sparse
from sklearn.metrics import log_loss

from .config import MODEL_DIR, REPORT_DIR
from .model import (
    _football_pipelines,
    _validation_split,
    apply_calibrator,
    betting_metrics,
    choose_calibration,
    choose_edge_threshold,
    evaluate,
    feature_columns,
    fit_calibrator,
    make_preprocessor,
    walk_forward as walk_forward_v3,
    with_market_logit,
)

CORRECTION_SCALES = (0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50)


@dataclass
class OffsetLogisticCorrection:
    """Logistic correction with the market logit fixed as an offset.

    final_logit = market_logit + intercept + X @ beta

    The coefficient on the market logit is therefore fixed at exactly 1.0.
    """

    C: float = 0.25
    max_iter: int = 500
    coef_: np.ndarray | None = None
    intercept_: float = 0.0
    success_: bool = False

    def fit(self, X, y: np.ndarray, market_logit: np.ndarray) -> "OffsetLogisticCorrection":
        X = sparse.csr_matrix(X)
        y = np.asarray(y, dtype=float)
        market_logit = np.asarray(market_logit, dtype=float)
        n, p = X.shape

        if n == 0:
            raise ValueError("Cannot fit residual model on zero rows.")

        reg = 1.0 / (max(self.C, 1e-8) * n)

        def objective(theta: np.ndarray):
            intercept = theta[0]
            beta = theta[1:]
            z = market_logit + intercept + X.dot(beta)
            loss = np.mean(np.logaddexp(0.0, z) - y * z)
            loss += 0.5 * reg * float(beta @ beta)

            err = expit(z) - y
            grad_intercept = float(np.mean(err))
            grad_beta = np.asarray(X.T.dot(err)).ravel() / n + reg * beta
            grad = np.concatenate(([grad_intercept], grad_beta))
            return float(loss), grad

        initial = np.zeros(p + 1, dtype=float)
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter, "ftol": 1e-10},
        )
        self.intercept_ = float(result.x[0])
        self.coef_ = np.asarray(result.x[1:], dtype=float)
        self.success_ = bool(result.success)
        return self

    def predict_delta(self, X) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("OffsetLogisticCorrection has not been fitted.")
        X = sparse.csr_matrix(X)
        return self.intercept_ + np.asarray(X.dot(self.coef_)).ravel()

    def predict_proba(self, X, market_logit: np.ndarray, scale: float = 1.0) -> np.ndarray:
        delta = self.predict_delta(X)
        return expit(np.asarray(market_logit, dtype=float) + float(scale) * delta)


def choose_correction_scale(
    y: pd.Series,
    market_logit: np.ndarray,
    delta: np.ndarray,
) -> tuple[float, dict[float, float]]:
    y_arr = np.asarray(y, dtype=int)
    market_logit = np.asarray(market_logit, dtype=float)
    delta = np.asarray(delta, dtype=float)

    scores: dict[float, float] = {}
    for scale in CORRECTION_SCALES:
        p = expit(market_logit + scale * delta)
        scores[scale] = float(log_loss(y_arr, p, labels=[0, 1]))
    selected = min(scores, key=scores.get)
    return float(selected), scores


def _football_reference_probabilities(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> np.ndarray:
    features = numeric + categorical
    football_model = dict(_football_pipelines(numeric, categorical))["logistic"]
    football_model.fit(train[features], train.over_2_5)

    p_val_raw = football_model.predict_proba(val[features])[:, 1]
    calibration, calibrator, _ = choose_calibration(val, p_val_raw)

    p_test_raw = football_model.predict_proba(test[features])[:, 1]
    return apply_calibrator(calibration, calibrator, p_test_raw)


def _all_prediction_rows(
    test: pd.DataFrame,
    probability: np.ndarray,
    football_probability: np.ndarray,
    delta_logit: np.ndarray,
    correction_scale: float,
    edge_threshold: float | None,
    val_season: str,
    test_season: str,
) -> pd.DataFrame:
    out = pd.DataFrame({
        "date": test.date.to_numpy(),
        "season": test.season.astype(str).to_numpy(),
        "league": test.league.to_numpy(),
        "home_team": test.home_team.to_numpy(),
        "away_team": test.away_team.to_numpy(),
        "home_goals": test.home_goals.to_numpy(),
        "away_goals": test.away_goals.to_numpy(),
        "actual_over25": test.over_2_5.astype(int).to_numpy(),
        "opening_market": test.market_prob_open_over.to_numpy(),
        "closing_market": test.market_prob_close_over.to_numpy(),
        "opening_odds_over": test.odds_over_25_open.to_numpy(),
        "opening_odds_under": test.odds_under_25_open.to_numpy(),
        "closing_odds_over": test.odds_over_25_close.to_numpy(),
        "closing_odds_under": test.odds_under_25_close.to_numpy(),
        "poisson_probability": test.poisson_prob_over25.to_numpy(),
        "football_probability": football_probability,
        "residual_delta_logit": delta_logit,
        "correction_scale": correction_scale,
        "residual_probability": probability,
        "selected_threshold": edge_threshold,
        "val_season": val_season,
        "test_season": test_season,
    })

    out["edge_open"] = out.residual_probability - out.opening_market
    out["abs_edge_open"] = out.edge_open.abs()
    out["closing_move_over"] = out.closing_market - out.opening_market

    if edge_threshold is None or pd.isna(edge_threshold):
        out["bet_side"] = "none"
    else:
        out["bet_side"] = np.where(
            out.edge_open >= edge_threshold,
            "over",
            np.where(out.edge_open <= -edge_threshold, "under", "none"),
        )

    over_odds = test.odds_over_25_open.to_numpy()
    under_odds = test.odds_under_25_open.to_numpy()
    out["bet_odds"] = np.where(
        out.bet_side.eq("over"),
        over_odds,
        np.where(out.bet_side.eq("under"), under_odds, np.nan),
    )

    won = np.where(
        out.bet_side.eq("over"),
        out.actual_over25.eq(1),
        np.where(out.bet_side.eq("under"), out.actual_over25.eq(0), False),
    )
    out["profit"] = np.where(
        out.bet_side.eq("none"),
        np.nan,
        np.where(won, out.bet_odds - 1.0, -1.0),
    )

    out["clv_prob"] = np.where(
        out.bet_side.eq("over"),
        out.closing_market - out.opening_market,
        np.where(
            out.bet_side.eq("under"),
            out.opening_market - out.closing_market,
            np.nan,
        ),
    )
    return out


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    valid = a.notna() & b.notna()
    if valid.sum() < 3:
        return np.nan
    if a.loc[valid].nunique() < 2 or b.loc[valid].nunique() < 2:
        return np.nan
    return float(a.loc[valid].corr(b.loc[valid]))


def train_residual_once(
    df: pd.DataFrame,
    train_seasons: list[str],
    val_season: str,
    test_season: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = with_market_logit(
        df[df.season.astype(str).isin(train_seasons)].copy().sort_values("date")
    )
    val = with_market_logit(
        df[df.season.astype(str).eq(str(val_season))].copy().sort_values("date")
    )
    test = with_market_logit(
        df[df.season.astype(str).eq(str(test_season))].copy().sort_values("date")
    )

    for name, frame in (("train", train), ("val", val), ("test", test)):
        filtered = frame[
            (frame.home_matches_seen >= 5)
            & (frame.away_matches_seen >= 5)
            & frame.market_logit_open.notna()
        ].copy()
        if name == "train":
            train = filtered
        elif name == "val":
            val = filtered
        else:
            test = filtered

    numeric, categorical = feature_columns(train, include_market=False)
    features = numeric + categorical

    preprocessor = make_preprocessor(numeric, categorical)
    X_train = preprocessor.fit_transform(train[features])
    X_val = preprocessor.transform(val[features])
    X_test = preprocessor.transform(test[features])

    residual = OffsetLogisticCorrection(C=0.25, max_iter=600)
    residual.fit(
        X_train,
        train.over_2_5.to_numpy(),
        train.market_logit_open.to_numpy(),
    )

    delta_val = residual.predict_delta(X_val)
    delta_test = residual.predict_delta(X_test)

    split = _validation_split(val)
    correction_scale, scale_scores = choose_correction_scale(
        val.over_2_5.iloc[:split],
        val.market_logit_open.iloc[:split].to_numpy(),
        delta_val[:split],
    )

    p_val_pick = expit(
        val.market_logit_open.iloc[split:].to_numpy()
        + correction_scale * delta_val[split:]
    )
    edge_threshold, _ = choose_edge_threshold(
        val.iloc[split:].copy(),
        p_val_pick,
    )

    p_test = expit(
        test.market_logit_open.to_numpy()
        + correction_scale * delta_test
    )
    football_probability = _football_reference_probabilities(
        train, val, test, numeric, categorical
    )

    metrics = evaluate(test.over_2_5, p_test)
    metrics.update(betting_metrics(test, p_test, edge_threshold))

    closing_move = test.market_prob_close_over - test.market_prob_open_over
    metrics.update({
        "model": "market_residual_logistic",
        "calibration": "fixed_market_offset",
        "val_cal_logloss": scale_scores[correction_scale],
        "edge_threshold": edge_threshold,
        "blend_weight": np.nan,
        "correction_scale": correction_scale,
        "mean_abs_correction": float(np.mean(np.abs(p_test - test.market_prob_open_over.to_numpy()))),
        "correction_close_corr": _safe_corr(
            pd.Series(p_test - test.market_prob_open_over.to_numpy()),
            closing_move.reset_index(drop=True),
        ),
        "optimizer_success": residual.success_,
        "val_season": val_season,
        "test_season": test_season,
        "n_test": len(test),
        "market_coverage": 1.0,
    })

    predictions = _all_prediction_rows(
        test,
        p_test,
        football_probability,
        delta_test,
        correction_scale,
        edge_threshold,
        val_season,
        test_season,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "preprocessor": preprocessor,
        "residual_model": residual,
        "correction_scale": correction_scale,
        "edge_threshold": edge_threshold,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "train_seasons": train_seasons,
        "scale_selected_on": val_season,
        "evaluated_on": test_season,
    }, MODEL_DIR / "over25_market_residual.joblib")

    return pd.DataFrame([metrics]), predictions


def _subgroup_metric_row(
    group_type: str,
    group_value: str,
    frame: pd.DataFrame,
) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "group_type": group_type,
        "group_value": group_value,
        "n": int(len(frame)),
    }
    if frame.empty:
        return row

    valid_prob = frame.residual_probability.notna()
    if valid_prob.any():
        scored = frame.loc[valid_prob]
        ev = evaluate(scored.actual_over25, scored.residual_probability.to_numpy())
        row.update(ev)

    valid_open = frame.opening_market.notna()
    if valid_open.any():
        opened = frame.loc[valid_open]
        opening_ev = evaluate(opened.actual_over25, opened.opening_market.to_numpy())
        row["opening_brier"] = opening_ev["brier"]
        row["opening_log_loss"] = opening_ev["log_loss"]
        row["opening_roc_auc"] = opening_ev["roc_auc"]
        if "brier" in row:
            row["brier_gain_vs_opening"] = opening_ev["brier"] - float(row["brier"])
            row["logloss_gain_vs_opening"] = opening_ev["log_loss"] - float(row["log_loss"])

    bets = frame[frame.bet_side.ne("none") & frame.profit.notna()]
    row["bets"] = int(len(bets))
    row["roi"] = float(bets.profit.mean()) if len(bets) else np.nan
    row["avg_clv_prob"] = float(bets.clv_prob.mean()) if len(bets) else np.nan
    row["mean_abs_edge"] = float(frame.abs_edge_open.mean())
    row["correction_close_corr"] = _safe_corr(
        frame.residual_probability - frame.opening_market,
        frame.closing_move_over,
    )
    return row


def build_subgroup_report(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    rows.append(_subgroup_metric_row("all", "all", predictions))

    for league, frame in predictions.groupby("league"):
        rows.append(_subgroup_metric_row("league", str(league), frame))

    selected = predictions[predictions.bet_side.ne("none")].copy()
    for side, frame in selected.groupby("bet_side"):
        rows.append(_subgroup_metric_row("bet_side", str(side), frame))

    bins = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, np.inf]
    labels = ["0-2%", "2-4%", "4-6%", "6-8%", "8-10%", "10-15%", "15%+"]
    bucketed = predictions.copy()
    bucketed["edge_bucket"] = pd.cut(
        bucketed.abs_edge_open,
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    for bucket, frame in bucketed.groupby("edge_bucket", observed=True):
        rows.append(_subgroup_metric_row("abs_edge", str(bucket), frame))

    return pd.DataFrame(rows)


def walk_forward_residual(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = sorted(df.season.astype(str).unique())
    metrics = []
    predictions = []

    for test_idx in range(4, len(seasons)):
        test_season = seasons[test_idx]
        val_season = seasons[test_idx - 1]
        train_seasons = seasons[:test_idx - 1]
        fold_metrics, fold_predictions = train_residual_once(
            df, train_seasons, val_season, test_season
        )
        metrics.append(fold_metrics)
        predictions.append(fold_predictions)

    return (
        pd.concat(metrics, ignore_index=True),
        pd.concat(predictions, ignore_index=True),
    )


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    v3_report = walk_forward_v3(df)
    residual_report, predictions = walk_forward_residual(df)

    report = pd.concat([v3_report, residual_report], ignore_index=True, sort=False)
    subgroup_report = build_subgroup_report(predictions)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORT_DIR / "walk_forward_metrics.csv"
    predictions_path = REPORT_DIR / "walk_forward_predictions.csv"
    subgroup_path = REPORT_DIR / "v4_subgroup_metrics.csv"

    report.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    subgroup_report.to_csv(subgroup_path, index=False)

    print("\nWalk-forward results")
    print(report.sort_values(["test_season", "model"]).to_string(index=False))
    print(f"\nsaved -> {metrics_path}")
    print(f"saved -> {predictions_path}")
    print(f"saved -> {subgroup_path}")

    return report
