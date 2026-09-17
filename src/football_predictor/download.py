from __future__ import annotations

import argparse
from pathlib import Path
import requests

from .config import RAW_DIR, load_config

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    if len(response.content) < 100:
        raise RuntimeError(f"Suspiciously small response from {url}")
    destination.write_bytes(response.content)


def download_all(include_current: bool = False, force: bool = False) -> None:
    cfg = load_config()
    seasons = list(cfg["seasons"])
    if include_current:
        seasons.append(str(cfg["current_season"]))

    for season in seasons:
        for league in cfg["leagues"]:
            dest = RAW_DIR / f"{season}_{league}.csv"
            if dest.exists() and not force:
                print(f"skip  {dest.name}")
                continue
            url = BASE_URL.format(season=season, league=league)
            try:
                download_file(url, dest)
                print(f"saved {dest.name}")
            except Exception as exc:
                print(f"FAIL  {season} {league}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-current", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_all(args.include_current, args.force)


if __name__ == "__main__":
    main()
