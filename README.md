# Football Predictor V2 — Over/Under 2.5 Goals

A leakage-safe football probability model for predicting whether a match will finish with **3+ total goals**.

## V2 design

- Target: `Over 2.5 goals`
- Leagues: Premier League, Serie A, La Liga, Bundesliga, Ligue 1
- Historical source: Football-Data.co.uk CSVs
- Baselines: league historical rate + logistic regression
- Main model: LightGBM
- Calibration: raw vs Platt/sigmoid vs isotonic, selected chronologically inside the validation season
- Evaluation: walk-forward backtest
- Betting layer: two-sided Over/Under edge selection, flat-stake ROI, drawdown and closing-line movement

The model deliberately excludes bookmaker odds from the core predictive feature set. Odds are used *after prediction* to evaluate whether the model's estimated probability differs from the market.

## Why this avoids leakage

Features are created chronologically. For every match, each team's rolling statistics are read **before** that match is added to its history. Therefore a game's result, shots, or goals cannot appear in its own predictors.

## Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Download completed historical seasons:

```bash
python run.py download
```

Build leakage-safe features:

```bash
python run.py features
```

Run walk-forward backtests:

```bash
python run.py backtest
```

The report is written to:

```text
reports/walk_forward_metrics.csv
```

## Current feature groups

- Rolling goals for / against over 5 and 10 matches
- Rolling shots and shots-on-target over 5 and 10 matches
- Rolling Over-2.5 frequency
- Rolling points per match
- Days of rest
- Number of prior matches seen
- League historical goal average available before kickoff
- Simple Poisson-style expected total-goals feature
- League identity

## Backtest protocol

For a test season `T`:

1. train on all seasons before `T-1`
2. calibrate on season `T-1`
3. evaluate once on `T`
4. move the window forward

This is intentionally stricter than a random train/test split.

## Next milestones

1. Run the first backtest and inspect calibration by probability bucket.
2. Add Elo / attack / defence ratings.
3. Add home-only and away-only rolling form.
4. Add current fixtures and automatic daily prediction output.
5. Store odds snapshots so closing-line and pre-match value can be measured properly.
6. Add API-Football only after the historical model proves useful.

## Disclaimer

This is a statistical research project. Good historical backtests do not guarantee profitable future betting results.
