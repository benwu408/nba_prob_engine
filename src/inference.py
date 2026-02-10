"""
Run the trained win-probability model on a single game (e.g. frontend 2025-26 data).

- load_model(): load model + scaler from joblib.
- run_game_win_prob(): given game_id and paths, return play-by-play with prob_home_win at each event.
"""
from pathlib import Path

import pandas as pd

from .config import FRONTEND_PARSED_DIR, PARSED_DIR
from .game_state import apply_event, initial_state
from .train import DEFAULT_MODEL_PATH, FEATURE_COLUMNS

try:
    import joblib
except ImportError:
    joblib = None


def load_model(path: Path | None = None):
    """Load model and scaler from joblib. Returns (model, scaler)."""
    if joblib is None:
        raise ImportError("joblib is required; pip install joblib")
    path = Path(path or DEFAULT_MODEL_PATH)
    bundle = joblib.load(path)
    return bundle["model"], bundle["scaler"]


def run_game_win_prob(
    game_id: str,
    parsed_dir: Path | None = None,
    game_elos_path: Path | None = None,
    model_path: Path | None = None,
) -> list[dict]:
    """
    Replay one game and return a row per event with state + prob_home_win.

    game_id: 10-char game id (e.g. 0022500001).
    parsed_dir: directory containing games/<game_id>_events.csv (default: FRONTEND_PARSED_DIR).
    game_elos_path: CSV with game_id, home_elo, away_elo (default: parsed_dir / game_elos.csv).
    model_path: path to win_prob_model.joblib (default: training model path).

    Returns list of dicts with: event_num, period, time_remaining_sec, home_score, away_score,
    score_diff, possession, event_type, description, prob_home_win.
    """
    parsed_dir = Path(parsed_dir or FRONTEND_PARSED_DIR)
    game_id = str(game_id).zfill(10)
    events_path = parsed_dir / "games" / f"{game_id}_events.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"Events not found: {events_path}")

    events_df = pd.read_csv(events_path).sort_values("event_num")
    if events_df.empty:
        return []

    home_elo = 1500.0
    away_elo = 1500.0
    if game_elos_path is None:
        game_elos_path = parsed_dir / "game_elos.csv"
    if Path(game_elos_path).exists():
        elo_df = pd.read_csv(game_elos_path, dtype={"game_id": str})
        elo_df["game_id"] = elo_df["game_id"].str.zfill(10)
        row = elo_df[elo_df["game_id"] == game_id]
        if not row.empty:
            home_elo = float(row.iloc[0]["home_elo"])
            away_elo = float(row.iloc[0]["away_elo"])

    model, scaler = load_model(path=model_path)
    state = initial_state()
    out = []
    for _, row in events_df.iterrows():
        state = apply_event(state, row.to_dict())
        feat = state.to_feature_dict()
        feat["home_elo"] = home_elo
        feat["away_elo"] = away_elo
        X = scaler.transform(pd.DataFrame([feat])[FEATURE_COLUMNS])
        prob = float(model.predict_proba(X)[0, 1])
        out.append({
            "event_num": int(row["event_num"]),
            "period": int(row["period"]),
            "time_remaining_sec": float(row["time_remaining_sec"]),
            "home_score": int(row["home_score"]),
            "away_score": int(row["away_score"]),
            "score_diff": state.score_diff,
            "possession": state.possession,
            "event_type": str(row.get("event_type", "")),
            "description": str(row.get("description", "")),
            "prob_home_win": round(prob, 4),
        })
    return out
