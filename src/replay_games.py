"""
Replay parsed games to build (state, label) training data.

- replay_one_game(events_df): yields (feature_dict, label) after each event.
- replay_all_games(parsed_dir, output_path, ...): replays all game CSVs and writes training_dataset.csv.
  If elo_df is provided (game_id, home_elo, away_elo), those columns are merged in.
"""
from pathlib import Path

import pandas as pd

from .config import PARSED_DIR
from .game_state import apply_event, initial_state
from .elo import ELO_INITIAL


def replay_one_game(events_df: pd.DataFrame):
    """
    Replay one game: start from initial state, apply each event in order, yield (features, label).

    events_df: DataFrame with event_num, period, time_remaining_sec, home_score, away_score, possession.
    Label is derived from the game outcome (home wins = 1) using the last event's scores.
    Yields: (dict of state features, label_home_win, event_num).
    """
    if events_df.empty:
        return
    events_df = events_df.sort_values("event_num")
    last = events_df.iloc[-1]
    label = 1 if int(last["home_score"]) > int(last["away_score"]) else 0

    state = initial_state()
    for _, row in events_df.iterrows():
        state = apply_event(state, row.to_dict())
        yield state.to_feature_dict(), label, int(row["event_num"])


def replay_all_games(
    parsed_dir: Path | None = None,
    output_path: Path | None = None,
    *,
    every_n: int = 1,
    limit_games: int | None = None,
    elo_df: pd.DataFrame | None = None,
) -> int:
    """
    Replay all games in data/parsed/games/*_events.csv and write training rows to a CSV.

    parsed_dir: directory containing the "games" subdir (default: config PARSED_DIR).
    output_path: path for output CSV (default: parsed_dir / "training_dataset.csv").
    every_n: emit a row every N events (1 = every event; 10 = every 10th event).
    limit_games: if set, only replay this many games (for testing).
    elo_df: optional DataFrame with game_id, home_elo, away_elo (pre-game ELO per season).
            If provided, these columns are merged in; missing games get ELO_INITIAL.

    Returns the number of training rows written.
    """
    parsed_dir = Path(parsed_dir or PARSED_DIR)
    output_path = Path(output_path or parsed_dir / "training_dataset.csv")
    games_dir = parsed_dir / "games"
    if not games_dir.is_dir():
        raise FileNotFoundError(f"Games directory not found: {games_dir}. Run parse step first.")

    paths = sorted(games_dir.glob("*_events.csv"))
    if limit_games is not None:
        paths = paths[:limit_games]

    rows = []
    n_games = len(paths)
    for idx, path in enumerate(paths, 1):
        if idx % 100 == 0 or idx == n_games:
            print(f"  Replaying game {idx}/{n_games}: {path.stem}", flush=True)
        df = pd.read_csv(path)
        game_id = path.stem.replace("_events", "")
        game_id = str(game_id).zfill(10)
        for i, (feat, label, event_num) in enumerate(replay_one_game(df)):
            if (i + 1) % every_n != 0:
                continue
            row = {"game_id": game_id, "event_num": event_num, "label_home_win": label, **feat}
            rows.append(row)

    if not rows:
        print("No rows produced.")
        return 0

    out = pd.DataFrame(rows)
    if elo_df is not None and not elo_df.empty and "home_elo" in elo_df.columns and "away_elo" in elo_df.columns:
        elo = elo_df[["game_id", "home_elo", "away_elo"]].copy()
        elo["game_id"] = elo["game_id"].astype(str).str.zfill(10)
        out = out.merge(elo, on="game_id", how="left")
        out["home_elo"] = out["home_elo"].fillna(ELO_INITIAL)
        out["away_elo"] = out["away_elo"].fillna(ELO_INITIAL)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} training rows to {output_path}")
    return len(out)
