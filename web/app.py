"""
Flask API for the win-probability frontend.

- GET /api/games?team=XXX  — list games (optional filter by team abbrev)
- GET /api/game/<game_id>/win_prob — play-by-play win prob for one game
- GET / — serve the frontend
"""
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

# Add project root so we can import src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FRONTEND_PARSED_DIR, FRONTEND_RAW_DIR
from src.inference import run_game_win_prob

app = Flask(__name__, static_folder="static", static_url_path="")


def _manifest_path() -> Path:
    p = FRONTEND_RAW_DIR / "games_manifest.csv"
    if not p.exists():
        return None
    return p


@app.route("/api/games")
def list_games():
    path = _manifest_path()
    if path is None:
        return jsonify({"error": "No frontend manifest. Run run_frontend_data.py first."}), 503
    df = pd.read_csv(path, dtype={"game_id": str})
    df["game_id"] = df["game_id"].str.zfill(10)
    team = (request.args.get("team") or "").strip().upper()
    if team:
        mask = (df["home_team_abbrev"].str.upper() == team) | (df["away_team_abbrev"].str.upper() == team)
        df = df.loc[mask]
    # Sort by date desc (newest first)
    df = df.sort_values("game_date", ascending=False).reset_index(drop=True)
    games = []
    for _, row in df.iterrows():
        games.append({
            "game_id": row["game_id"],
            "game_date": str(row["game_date"]),
            "home_team": row["home_team_abbrev"],
            "away_team": row["away_team_abbrev"],
            "pts_home": int(row["pts_home"]) if pd.notna(row.get("pts_home")) else None,
            "pts_away": int(row["pts_away"]) if pd.notna(row.get("pts_away")) else None,
            "wl_home": str(row.get("wl_home", "")),
        })
    return jsonify({"games": games})


@app.route("/api/game/<game_id>/win_prob")
def game_win_prob(game_id: str):
    path = _manifest_path()
    if path is None:
        return jsonify({"error": "No frontend data. Run run_frontend_data.py first."}), 503
    try:
        rows = run_game_win_prob(
            game_id,
            parsed_dir=FRONTEND_PARSED_DIR,
            game_elos_path=FRONTEND_PARSED_DIR / "game_elos.csv",
        )
        return jsonify({"game_id": game_id, "events": rows})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def static_file(path):
    return send_from_directory(app.static_folder, path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
