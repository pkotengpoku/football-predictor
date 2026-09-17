from __future__ import annotations

import argparse
import pandas as pd

from .config import PROCESSED_DIR, REPORT_DIR
from .download import download_all
from .features import build_and_save
from .model import walk_forward


def main() -> None:
    parser = argparse.ArgumentParser(prog="football-predictor")
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download")
    dl.add_argument("--include-current", action="store_true")
    dl.add_argument("--force", action="store_true")

    sub.add_parser("features")
    sub.add_parser("backtest")

    args = parser.parse_args()
    if args.command == "download":
        download_all(args.include_current, args.force)
    elif args.command == "features":
        build_and_save()
    elif args.command == "backtest":
        df = pd.read_parquet(PROCESSED_DIR / "matches_features.parquet")
        report = walk_forward(df)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORT_DIR / "walk_forward_metrics.csv"
        report.to_csv(out, index=False)
        print(report.to_string(index=False))
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
