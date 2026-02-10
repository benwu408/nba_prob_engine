#!/usr/bin/env python3
"""
Step 0 + data prep: Fetch historical NBA play-by-play and parse into a dataset.

Usage:
  python run_pipeline.py                    # default season (2023-24), all games
  python run_pipeline.py --season 2022-23
  python run_pipeline.py --all-seasons      # fetch 2022-23, 2023-24, 2024-25
  python run_pipeline.py --limit 20        # first 20 games only (for testing)
  python run_pipeline.py --fetch-only       # only download raw pbp
  python run_pipeline.py --parse-only       # only parse existing raw data
"""
import argparse
from pathlib import Path

from src.config import RAW_DIR
from src.fetch import fetch_all, fetch_all_seasons, get_games
from src.parse_events import load_game_manifest, parse_all_games


def main():
    p = argparse.ArgumentParser(description="Fetch NBA play-by-play and parse to dataset")
    p.add_argument("--season", default="2023-24", help="Season e.g. 2023-24 (ignored if --all-seasons)")
    p.add_argument("--all-seasons", action="store_true", help="Fetch 2022-23, 2023-24, 2024-25 (seasons supported by live endpoint)")
    p.add_argument("--limit", type=int, default=None, help="Max games per season (for testing)")
    p.add_argument("--fetch-only", action="store_true", help="Only fetch raw data, do not parse")
    p.add_argument("--parse-only", action="store_true", help="Only parse existing raw data")
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Directory for raw CSV files")
    args = p.parse_args()

    if not args.parse_only:
        if args.all_seasons:
            print("Fetching games for all seasons (2022-23, 2023-24, 2024-25; live endpoint)" + (f", limit={args.limit} per season" if args.limit else ""))
            print("(First step: getting game list from API — may take a few seconds.)", flush=True)
            games_df, failed = fetch_all_seasons(
                output_dir=args.raw_dir,
                limit_per_season=args.limit,
            )
        else:
            print(f"Fetching games for season {args.season}" + (f" (limit={args.limit})" if args.limit else ""))
            print("(First step: getting game list from API — may take a few seconds.)", flush=True)
            games_df, failed = fetch_all(
                season=args.season,
                output_dir=args.raw_dir,
                limit=args.limit,
            )
        print(f"Games in manifest: {len(games_df)}")
        if failed:
            print(f"Failed to fetch pbp for {len(failed)} game(s): {failed[:5]}{'...' if len(failed) > 5 else ''}")
        if args.fetch_only:
            return

    manifest = load_game_manifest(args.raw_dir)
    if manifest.empty:
        print("No games_manifest.csv found. Run without --parse-only first.")
        return

    raw_count = sum(1 for f in args.raw_dir.glob("*.csv") if f.name != "games_manifest.csv")
    print(f"\nParsing {raw_count} raw play-by-play files...")
    df = parse_all_games(raw_dir=args.raw_dir)
    if df.empty:
        print("No events parsed.")
        return
    print(f"Parsed {len(df)} events from {df['game_id'].nunique()} games.")
    print(f"Dataset saved to data/parsed/events_dataset.csv")
    print(f"Sample event types: {df['event_type'].value_counts().head(8).to_dict()}")


if __name__ == "__main__":
    main()
