"""
Parse raw nba_api play-by-play into normalized event sequences and dataset format.

- parse_pbp(raw_df, game_id, ...): raw pbp DataFrame -> list of event dicts
- parse_all_games(raw_dir, parsed_dir): all raw CSVs -> events_dataset.csv + per-game CSVs
- load_game_manifest(raw_dir): read games_manifest.csv (game_id as zero-padded string)

Event types: made_2, made_3, miss_2, miss_3, free_throw_made, free_throw_miss,
turnover, foul, rebound, period_start, period_end, timeout, jump_ball, etc.
Each event has: period, time_remaining_sec, home_score, away_score, possession,
event_type, points_scored, description.
"""
import re
from pathlib import Path

import pandas as pd

from .config import PARSED_DIR, RAW_DIR

# nba_api EVENTMSGTYPE (from their docs)
EVENT_TYPE_MAP = {
    1: "field_goal_made",
    2: "field_goal_missed",
    3: "free_throw",
    4: "rebound",
    5: "turnover",
    6: "foul",
    7: "violation",
    8: "substitution",
    9: "timeout",
    10: "jump_ball",
    11: "ejection",
    12: "period_start",
    13: "period_end",
    18: "instant_replay",
}


def _parse_clock(pctimestring: str) -> float:
    """Convert PCTIMESTRING (e.g. '11:25' or '0:00') to seconds remaining in period."""
    if pd.isna(pctimestring) or not str(pctimestring).strip():
        return 0.0
    s = str(pctimestring).strip()
    parts = s.split(":")
    if len(parts) != 2:
        return 0.0
    try:
        minutes = int(parts[0])
        seconds = float(parts[1]) if "." in parts[1] else int(parts[1])
        return minutes * 60 + seconds
    except (ValueError, TypeError):
        return 0.0


def _parse_score(score_str) -> tuple[int | None, int | None]:
    """Parse SCORE string '100 - 98' -> (100, 98). Returns (None, None) if invalid."""
    if pd.isna(score_str) or not str(score_str).strip():
        return None, None
    s = str(score_str).strip()
    parts = re.split(r"\s*-\s*", s, maxsplit=1)
    if len(parts) != 2:
        return None, None
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except (ValueError, TypeError):
        return None, None


def _infer_possession(row: pd.Series) -> str | None:
    """Infer which team had possession from HOMEDESCRIPTION / VISITORDESCRIPTION."""
    if pd.notna(row.get("HOMEDESCRIPTION")) and str(row["HOMEDESCRIPTION"]).strip():
        return "home"
    if pd.notna(row.get("VISITORDESCRIPTION")) and str(row["VISITORDESCRIPTION"]).strip():
        return "away"
    return None


def _points_from_description(msg_type: int, home_desc, away_desc) -> int:
    """Infer points scored this play from description (2pt, 3pt, FT)."""
    desc = home_desc if pd.notna(home_desc) else away_desc
    if pd.isna(desc):
        return 0
    desc = str(desc)
    if msg_type == 1:  # made FG
        if "(3 PTS)" in desc or "3PT" in desc and "MISS" not in desc:
            return 3
        return 2
    if msg_type == 3:  # free throw
        if "Free Throw" in desc and "MISS" not in desc:
            return 1
    return 0


def _refine_event_type(row: pd.Series, base_type: str) -> str:
    """Refine event type (e.g. field_goal_made -> made_2 / made_3)."""
    msg_type = row.get("EVENTMSGTYPE")
    home_desc = row.get("HOMEDESCRIPTION") or ""
    away_desc = row.get("VISITORDESCRIPTION") or ""
    desc = str(home_desc) + " " + str(away_desc)

    if base_type == "field_goal_made":
        return "made_3" if "(3 PTS)" in desc or ("3PT" in desc and "MISS" not in desc) else "made_2"
    if base_type == "field_goal_missed":
        return "miss_3" if "3PT" in desc else "miss_2"
    if base_type == "free_throw":
        if "MISS" in desc:
            return "free_throw_miss"
        return "free_throw_made"
    if base_type == "period_start":
        return "period_start"
    if base_type == "period_end":
        return "period_end"
    if base_type == "timeout":
        return "timeout"
    if base_type == "jump_ball":
        return "jump_ball"
    if base_type == "turnover":
        return "turnover"
    if base_type == "foul":
        return "foul"
    if base_type == "rebound":
        return "rebound"
    if base_type in ("substitution", "violation", "ejection", "instant_replay"):
        return base_type
    return base_type or "other"


def parse_pbp(
    raw_df: pd.DataFrame,
    game_id: str,
    *,
    home_team_id: int | None = None,
    away_team_id: int | None = None,
) -> list[dict]:
    """
    Convert raw play-by-play DataFrame to list of normalized event dicts.

    Each event has:
        game_id, event_num, period, time_remaining_sec, home_score, away_score,
        possession (home|away|None), event_type, points_scored, description
    Score is carried forward when not present in the raw row.
    """
    if raw_df.empty:
        return []

    events = []
    home_score, away_score = 0, 0

    for idx, row in raw_df.iterrows():
        event_num = row.get("EVENTNUM", idx)
        period = int(row.get("PERIOD", 1))
        time_sec = _parse_clock(row.get("PCTIMESTRING"))
        hs, aw = _parse_score(row.get("SCORE"))
        if hs is not None and aw is not None:
            home_score, away_score = hs, aw
        possession = _infer_possession(row)
        msg_type = row.get("EVENTMSGTYPE")
        base_type = EVENT_TYPE_MAP.get(int(msg_type) if pd.notna(msg_type) else None, "other")
        event_type = _refine_event_type(row, base_type)
        points = _points_from_description(
            int(msg_type) if pd.notna(msg_type) else 0,
            row.get("HOMEDESCRIPTION"),
            row.get("VISITORDESCRIPTION"),
        )
        desc = (row.get("HOMEDESCRIPTION") or row.get("VISITORDESCRIPTION") or row.get("NEUTRALDESCRIPTION") or "")

        events.append({
            "game_id": game_id,
            "event_num": event_num,
            "period": period,
            "time_remaining_sec": time_sec,
            "home_score": home_score,
            "away_score": away_score,
            "possession": possession,
            "event_type": event_type,
            "points_scored": points,
            "description": str(desc)[:200] if pd.notna(desc) else "",
        })

    return events


def load_game_manifest(raw_dir: Path | None = None) -> pd.DataFrame:
    """Load games_manifest.csv from raw dir. Empty DataFrame if missing."""
    raw_dir = raw_dir or RAW_DIR
    path = Path(raw_dir) / "games_manifest.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"game_id": str})
    df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    return df


def parse_all_games(
    raw_dir: Path | None = None,
    parsed_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Parse all raw play-by-play files in raw_dir and build a single dataset.

    Discovers every raw CSV in raw_dir (not just manifest) so every game with
    raw data gets parsed. Uses games_manifest.csv for final scores when
    available; otherwise infers from the last event of the game.
    """
    raw_dir = Path(raw_dir or RAW_DIR)
    parsed_dir = Path(parsed_dir or PARSED_DIR)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    # All raw game CSVs (exclude manifest)
    raw_files = [f for f in raw_dir.glob("*.csv") if f.name != "games_manifest.csv"]
    if not raw_files:
        return pd.DataFrame()

    print(f"  Found {len(raw_files)} raw game files to parse.", flush=True)
    manifest = load_game_manifest(raw_dir)
    manifest_by_id = {}
    if not manifest.empty:
        for _, row in manifest.iterrows():
            gid = str(row["game_id"]).zfill(10)
            manifest_by_id[gid] = row

    rows = []
    n_files = len(raw_files)
    for idx, path in enumerate(sorted(raw_files), 1):
        gid = path.stem
        if len(gid) != 10:
            gid = gid.zfill(10)
        if idx % 50 == 0 or idx == n_files:
            print(f"  Parsing game {idx}/{n_files}: {gid}", flush=True)
        raw_df = pd.read_csv(path)
        game_row = manifest_by_id.get(gid)
        events = parse_pbp(
            raw_df,
            gid,
            home_team_id=int(game_row["home_team_id"]) if game_row is not None and pd.notna(game_row.get("home_team_id")) else None,
            away_team_id=int(game_row["away_team_id"]) if game_row is not None and pd.notna(game_row.get("away_team_id")) else None,
        )
        if not events:
            continue
        # Use manifest for final score/label if present; else infer from last event
        if game_row is not None and pd.notna(game_row.get("pts_home")) and pd.notna(game_row.get("pts_away")):
            pts_home_final = int(game_row["pts_home"])
            pts_away_final = int(game_row["pts_away"])
        else:
            last = events[-1]
            pts_home_final = int(last["home_score"])
            pts_away_final = int(last["away_score"])
        home_win = 1 if pts_home_final > pts_away_final else 0
        for e in events:
            e["label_home_win"] = home_win
            e["pts_home_final"] = pts_home_final
            e["pts_away_final"] = pts_away_final
            rows.append(e)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    # Save combined dataset
    out_path = parsed_dir / "events_dataset.csv"
    out.to_csv(out_path, index=False)

    # Also save per-game JSON/CSV for replay use (optional: one file per game)
    games_out = parsed_dir / "games"
    games_out.mkdir(exist_ok=True)
    for gid in out["game_id"].unique():
        sub = out[out["game_id"] == gid].drop(columns=["label_home_win", "pts_home_final", "pts_away_final"])
        sub.to_csv(games_out / f"{gid}_events.csv", index=False)

    return out
