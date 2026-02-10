#!/usr/bin/env python3
"""
Replay parsed games to build the training dataset (state → label for win probability).

Prerequisites:
  - You must have already run the parse step so that data/parsed/games/*_events.csv exist.
  - From project root: python run_pipeline.py --parse-only   (or run full pipeline first)

What to do when running this script:
  1. From project root, run: python run_replay.py
  2. Wait while it replays each game (progress every 100 games). No input needed.
  3. Output is written to data/parsed/training_dataset.csv (features + label_home_win).

Usage:
  python run_replay.py                    # replay all games, one row per event
  python run_replay.py --every-n 10       # sample every 10th event (smaller file)
  python run_replay.py --limit 50         # test run: only 50 games
  python run_replay.py --output path.csv  # custom output path
"""
import argparse
from pathlib import Path

from src.config import PARSED_DIR, RAW_DIR
from src.elo import compute_elo_per_game
from src.replay_games import replay_all_games


def main():
    p = argparse.ArgumentParser(
        description="Replay parsed games to build (state, label) training dataset"
    )
    p.add_argument(
        "--parsed-dir",
        type=Path,
        default=PARSED_DIR,
        help="Directory containing parsed/games (default: data/parsed)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <parsed-dir>/training_dataset.csv)",
    )
    p.add_argument(
        "--every-n",
        type=int,
        default=1,
        help="Emit a row every N events (default: 1 = every event)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Replay only this many games (for testing)",
    )
    p.add_argument(
        "--no-elo",
        action="store_true",
        help="Do not add ELO features (skip manifest / per-season ELO)",
    )
    args = p.parse_args()

    print("Replaying games to build training dataset...")
    if args.limit:
        print(f"  (limit: {args.limit} games)")
    if args.every_n > 1:
        print(f"  (sampling every {args.every_n} events)")
    elo_df = None
    if not args.no_elo:
        manifest_path = RAW_DIR / "games_manifest.csv"
        if manifest_path.exists():
            print("  Computing per-season ELO from manifest...")
            elo_df = compute_elo_per_game(manifest_path)
            print(f"  ELO computed for {len(elo_df)} games.")
        else:
            print("  (No games_manifest.csv; skipping ELO.)")
    print()

    replay_all_games(
        parsed_dir=args.parsed_dir,
        output_path=args.output,
        every_n=args.every_n,
        limit_games=args.limit,
        elo_df=elo_df,
    )


if __name__ == "__main__":
    main()
