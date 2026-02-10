"""
Fetch historical NBA games and play-by-play from nba_api.

- get_games(season): list of game_ids with metadata (home_team_id, away_team_id, etc.)
- fetch_playbyplay(game_id): raw play-by-play DataFrame for one game (stats API, then live fallback)
- fetch_playbyplay_live(game_id): fetch from cdn.nba.com (works when stats.nba.com returns empty)
- fetch_all(season, output_dir): save raw pbp CSV per game + games manifest
"""
import re
import time
from pathlib import Path

import pandas as pd
import requests

from nba_api.stats.endpoints import leaguegamefinder, playbyplay
from nba_api.stats.library.parameters import SeasonType

from .config import DEFAULT_SEASON, RAW_DIR, REQUEST_DELAY_SEC

# cdn.nba.com live play-by-play (works when stats.nba.com fails)
LIVE_PBP_URL = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
LIVE_REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://www.nba.com/",
}

# Custom headers for stats.nba.com (v1.1.0+: proxy, headers, timeout on every request).
# From nba_api Stats Examples: https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/examples.md
# We add Referer + x-nba-stats-* (often needed for playbyplay); Accept uses application/json for API.
NBA_REQUEST_HEADERS = {
    "Host": "stats.nba.com",
    "Connection": "keep-alive",
    "Cache-Control": "max-age=0",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://stats.nba.com/",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}


def get_games(
    season: str = DEFAULT_SEASON,
    season_type: str = SeasonType.regular,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Get all games for a season as a DataFrame with one row per game.

    Columns include: game_id, game_date, home_team_id, away_team_id,
    home_team_abbrev, away_team_abbrev, season_id, and final scores (pts_home, pts_away).
    """
    print(f"  Getting game list for {season}...", flush=True)
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable=season_type,
    )
    df = finder.get_data_frames()[0]
    print(f"  Got {len(df) // 2} games for {season}.", flush=True)

    # Two rows per game (one per team). Build one row per game with home/away.
    games_list = []
    for game_id, grp in df.groupby("GAME_ID"):
        rows = grp.to_dict("records")
        if len(rows) != 2:
            continue
        r0, r1 = rows[0], rows[1]
        # "XXX vs. YYY" => XXX is home; "XXX @ YYY" => XXX is away
        if " vs. " in r0["MATCHUP"]:
            home_row, away_row = r0, r1
        else:
            home_row, away_row = r1, r0
        games_list.append({
            "game_id": game_id,
            "game_date": home_row["GAME_DATE"],
            "season_id": home_row["SEASON_ID"],
            "home_team_id": home_row["TEAM_ID"],
            "away_team_id": away_row["TEAM_ID"],
            "home_team_abbrev": home_row["TEAM_ABBREVIATION"],
            "away_team_abbrev": away_row["TEAM_ABBREVIATION"],
            "pts_home": home_row["PTS"],
            "pts_away": away_row["PTS"],
            "wl_home": home_row["WL"],
        })
        if limit and len(games_list) >= limit:
            break

    out = pd.DataFrame(games_list)
    if out.empty:
        return out
    # Sort by date so we process in chronological order
    out = out.sort_values("game_date").reset_index(drop=True)
    return out


def _live_clock_to_pctimestring(clock: str) -> str:
    """Convert live clock 'PT11M08.00S' to stats format '11:08'."""
    if not clock or not isinstance(clock, str):
        return "0:00"
    m = re.match(r"PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?", clock)
    if not m:
        return "0:00"
    minutes = int(m.group(1) or 0)
    seconds = float(m.group(2) or 0)
    return f"{minutes}:{int(seconds):02d}"


def _live_action_type_to_eventmsgtype(action: dict) -> int:
    """Map cdn.nba.com actionType/subType/shotResult to EVENTMSGTYPE (1-13, 18)."""
    at = (action.get("actionType") or "").lower()
    sub = (action.get("subType") or "").lower()
    result = (action.get("shotResult") or "").lower()
    if at in ("period", "startperiod") or "period start" in (action.get("description") or "").lower():
        return 12  # period_start
    if at == "endperiod":
        return 13  # period_end
    if at == "jumpball":
        return 10
    if at == "violation":
        return 7
    if at == "timeout":
        return 9
    if at == "substitution":
        return 8
    if at == "foul":
        return 6
    if at == "turnover":
        return 5
    if at == "rebound":
        return 4
    if at in ("2pt", "3pt"):
        return 1 if result == "made" else 2
    if at == "freethrow":
        return 3
    if "block" in sub or "block" in at:
        return 2  # missed shot (block)
    return 0  # other


def fetch_playbyplay_live(game_id: str, timeout: int = 15) -> pd.DataFrame | None:
    """
    Fetch play-by-play from cdn.nba.com (live endpoint). Returns a DataFrame in the
    same column shape as stats.nba.com PlayByPlay so our parser can consume it.
    Use when stats.nba.com returns empty or fails.
    """
    game_id = str(game_id).zfill(10)
    url = LIVE_PBP_URL.format(game_id=game_id)
    try:
        r = requests.get(url, headers=LIVE_REQUEST_HEADERS, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError):
        return None
    game = data.get("game") or {}
    actions = game.get("actions") or []
    if not actions:
        return None
    # Infer home team: team that scores first to increase scoreHome
    home_team_id = None
    away_team_id = None
    for a in actions:
        sh = int(a.get("scoreHome") or 0)
        sa = int(a.get("scoreAway") or 0)
        tid = a.get("teamId")
        if tid and sh > 0 and home_team_id is None:
            home_team_id = tid
        if tid and sa > 0 and away_team_id is None:
            away_team_id = tid
        if home_team_id is not None and away_team_id is not None:
            break
    if home_team_id is None and actions:
        home_team_id = actions[0].get("possession") or actions[0].get("teamId")
    if away_team_id is None and actions:
        for a in actions:
            tid = a.get("teamId")
            if tid and tid != home_team_id:
                away_team_id = tid
                break
    rows = []
    score_home, score_away = 0, 0
    for a in actions:
        sh = int(a.get("scoreHome") or 0)
        sa = int(a.get("scoreAway") or 0)
        if sh > 0 or sa > 0:
            score_home, score_away = sh, sa
        tid = a.get("teamId")
        is_home = tid == home_team_id if home_team_id else False
        is_away = tid == away_team_id if away_team_id else False
        desc = (a.get("description") or "")[:500]
        home_desc = desc if is_home else None
        away_desc = desc if is_away else None
        if not is_home and not is_away and desc:
            away_desc = desc  # default to visitor if unknown
        score_str = f"{score_home} - {score_away}"
        margin = score_home - score_away
        scoremargin = str(margin) if margin != 0 else "TIE"
        rows.append({
            "GAME_ID": game_id,
            "EVENTNUM": a.get("actionNumber", len(rows) + 1),
            "EVENTMSGTYPE": _live_action_type_to_eventmsgtype(a),
            "EVENTMSGACTIONTYPE": 0,
            "PERIOD": int(a.get("period") or 1),
            "WCTIMESTRING": "",
            "PCTIMESTRING": _live_clock_to_pctimestring(a.get("clock") or ""),
            "HOMEDESCRIPTION": home_desc,
            "NEUTRALDESCRIPTION": None,
            "VISITORDESCRIPTION": away_desc,
            "SCORE": score_str,
            "SCOREMARGIN": scoremargin,
        })
    if not rows:
        return None
    return pd.DataFrame(rows)


def fetch_playbyplay(
    game_id: str,
    *,
    start_period: int = 1,
    end_period: int = 14,
    headers: dict | None = None,
    timeout: int = 20,
    try_stats_first: bool = False,
    stats_timeout: int = 5,
) -> pd.DataFrame | None:
    """
    Fetch raw play-by-play for one game. Returns DataFrame or None on failure.

    Uses cdn.nba.com (live) only by default — same as when we got 3k+ raw files.
    Set try_stats_first=True to try stats.nba.com first (then fall back to live).
    """
    game_id = str(game_id).zfill(10)
    if try_stats_first:
        try:
            pbp = playbyplay.PlayByPlay(
                game_id=game_id,
                start_period=start_period,
                end_period=end_period,
                headers=headers or NBA_REQUEST_HEADERS,
                timeout=stats_timeout,
            )
            df = pbp.get_data_frames()[0]
            if df is not None and not df.empty:
                return df
        except (KeyError, Exception):
            pass
    return fetch_playbyplay_live(game_id, timeout=timeout)


def fetch_all(
    season: str = DEFAULT_SEASON,
    season_type: str = SeasonType.regular,
    output_dir: Path | None = None,
    limit: int | None = None,
    delay_sec: float = REQUEST_DELAY_SEC,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Fetch all games for a season and save raw play-by-play to CSV per game.

    Returns:
        games_df: manifest of games (from get_games)
        failed_ids: list of game_ids for which pbp fetch failed
    """
    out = output_dir or RAW_DIR
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    games_df = get_games(season=season, season_type=season_type, limit=limit)
    if games_df.empty:
        return games_df, []

    failed = []
    to_fetch = [str(row["game_id"]).zfill(10) for _, row in games_df.iterrows() if not (out / f"{str(row['game_id']).zfill(10)}.csv").exists()]
    total = len(to_fetch)
    skipped = len(games_df) - total
    if skipped:
        print(f"  {skipped} games already on disk (skipping).", flush=True)
    if total == 0:
        print("  Nothing new to fetch.", flush=True)
    done = 0
    for i, row in games_df.iterrows():
        gid = str(row["game_id"]).zfill(10)
        path = out / f"{gid}.csv"
        if path.exists():
            continue
        done += 1
        df = fetch_playbyplay(gid)
        if df is None or df.empty:
            failed.append(gid)
            print(f"  [{done}/{total}] Failed {gid}", flush=True)
            continue
        df.to_csv(path, index=False)
        print(f"  [{done}/{total}] Fetched {gid} ({len(df)} events)", flush=True)
        if delay_sec > 0:
            time.sleep(delay_sec)

    # Merge with existing manifest so we never drop games that already have raw data
    manifest_path = out / "games_manifest.csv"
    existing = pd.DataFrame()
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path, dtype={"game_id": str})
        existing["game_id"] = existing["game_id"].astype(str).str.zfill(10)
    games_df["game_id"] = games_df["game_id"].astype(str).str.zfill(10)
    merged = pd.concat([existing, games_df], ignore_index=True).drop_duplicates(subset=["game_id"], keep="last")
    merged = merged.sort_values("game_date").reset_index(drop=True)
    merged.to_csv(manifest_path, index=False)
    return merged, failed


# Default season range for "all seasons" — live (cdn) only has recent seasons; older return 403
ALL_SEASONS_DEFAULT = [f"{y}-{str(y + 1)[2:]}" for y in range(2022, 2025)]  # 2022-23, 2023-24, 2024-25


def fetch_all_seasons(
    seasons: list[str] | None = None,
    season_type: str = SeasonType.regular,
    output_dir: Path | None = None,
    limit_per_season: int | None = None,
    delay_sec: float = REQUEST_DELAY_SEC,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Fetch games for multiple seasons and save raw play-by-play. Skips games that
    already have a CSV. Writes one combined games_manifest.csv at the end.

    Returns:
        games_df: combined manifest (all seasons)
        failed_ids: list of game_ids that failed to fetch
    """
    seasons = seasons or ALL_SEASONS_DEFAULT
    out = output_dir or RAW_DIR
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    all_games = []
    failed = []
    for season in seasons:
        print(f"\n--- Season {season} ---", flush=True)
        games_df = get_games(season=season, season_type=season_type, limit=limit_per_season)
        if games_df.empty:
            print(f"  No games returned, skipping.", flush=True)
            continue
        n_games = len(games_df)
        to_fetch = [str(row["game_id"]).zfill(10) for _, row in games_df.iterrows() if not (out / f"{str(row['game_id']).zfill(10)}.csv").exists()]
        n_to_fetch = len(to_fetch)
        print(f"  {n_games} games in list, {n_to_fetch} need fetching (rest already on disk).", flush=True)
        done = 0
        for i, row in games_df.iterrows():
            all_games.append(row)
            gid = str(row["game_id"]).zfill(10)
            path = out / f"{gid}.csv"
            if path.exists():
                continue
            done += 1
            df = fetch_playbyplay(gid)
            if df is None or df.empty:
                failed.append(gid)
                print(f"  [{done}/{n_to_fetch}] Failed {gid}", flush=True)
                continue
            df.to_csv(path, index=False)
            print(f"  [{done}/{n_to_fetch}] Fetched {gid} ({len(df)} events)", flush=True)
            if delay_sec > 0:
                time.sleep(delay_sec)
        print(f"  Season {season} done.", flush=True)

    combined = pd.DataFrame(all_games).drop_duplicates(subset=["game_id"], keep="first")
    if not combined.empty:
        combined = combined.sort_values("game_date").reset_index(drop=True)
        # Merge with existing manifest so we never drop games that already have raw data
        manifest_path = out / "games_manifest.csv"
        existing = pd.DataFrame()
        if manifest_path.exists():
            existing = pd.read_csv(manifest_path, dtype={"game_id": str})
            existing["game_id"] = existing["game_id"].astype(str).str.zfill(10)
        combined["game_id"] = combined["game_id"].astype(str).str.zfill(10)
        merged = pd.concat([existing, combined], ignore_index=True).drop_duplicates(subset=["game_id"], keep="last")
        merged = merged.sort_values("game_date").reset_index(drop=True)
        merged.to_csv(manifest_path, index=False)
        combined = merged
    return combined, failed
