#!/usr/bin/env python3
"""
Fetch and parse 2025-26 games for the frontend only. This data is NOT used for
training. Use it so you can pick any game from this season and run the existing
model play-by-play to see how win probability changes.

Output:
  data/frontend/raw/           — raw play-by-play CSVs + games_manifest.csv
  data/frontend/parsed/games/  — <game_id>_events.csv per game (for replay)
  data/frontend/parsed/game_elos.csv — game_id, home_elo, away_elo (for model input)

Usage:
  python run_frontend_data.py           # fetch all 2025-26 games so far, then parse + ELO
  python run_frontend_data.py --limit 20   # test: only 20 games
  python run_frontend_data.py --parse-only # only parse existing frontend raw (no fetch)
"""
import argparse
from pathlib import Path

from src.config import FRONTEND_PARSED_DIR, FRONTEND_RAW_DIR, FRONTEND_SEASON
from src.elo import compute_elo_per_game
from src.fetch import fetch_all
from src.parse_events import load_game_manifest, parse_all_games


def main():
    p = argparse.ArgumentParser(
        description="Fetch and parse current-season (2025-26) games for frontend; do not use for training."
    )
    p.add_argument("--limit", type=int, default=None, help="Max games to fetch (for testing)")
    p.add_argument("--parse-only", action="store_true", help="Only parse existing data in frontend/raw")
    args = p.parse_args()

    if not args.parse_only:
        print(f"Fetching {FRONTEND_SEASON} games to {FRONTEND_RAW_DIR}...")
        FRONTEND_RAW_DIR.mkdir(parents=True, exist_ok=True)
        games_df, failed = fetch_all(
            season=FRONTEND_SEASON,
            output_dir=FRONTEND_RAW_DIR,
            limit=args.limit,
        )
        print(f"Games in manifest: {len(games_df)}")
        if failed:
            print(f"Failed: {len(failed)} game(s)")
        if not games_df.empty and not args.parse_only:
            print()

    manifest_path = FRONTEND_RAW_DIR / "games_manifest.csv"
    if not manifest_path.exists():
        print("No games_manifest.csv in frontend/raw. Run without --parse-only first.")
        return

    print(f"Parsing from {FRONTEND_RAW_DIR} -> {FRONTEND_PARSED_DIR}...")
    parse_all_games(raw_dir=FRONTEND_RAW_DIR, parsed_dir=FRONTEND_PARSED_DIR)
    games_dir = FRONTEND_PARSED_DIR / "games"
    if games_dir.exists():
        n = len(list(games_dir.glob("*_events.csv")))
        print(f"Parsed {n} games -> {games_dir}")

    print("Computing per-season ELO for frontend games...")
    elo_df = compute_elo_per_game(manifest_path)
    elo_path = FRONTEND_PARSED_DIR / "game_elos.csv"
    elo_df.to_csv(elo_path, index=False)
    print(f"Saved {len(elo_df)} game ELOs -> {elo_path}")
    print("Done. Use data/frontend/parsed/ for the frontend (games + game_elos.csv).")


if __name__ == "__main__":
    main()
