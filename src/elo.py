"""
Per-season ELO from the games manifest.

- compute_elo_per_game(manifest): returns DataFrame with game_id, home_elo, away_elo
  (pre-game ELO for each team). ELO is computed separately per season_id; within
  a season, games are processed in game_date order and each team starts at 1500.
"""
from pathlib import Path

import pandas as pd

from .config import RAW_DIR

# Standard ELO: K factor and initial rating
ELO_K = 20
ELO_INITIAL = 1500


def _expected_score(elo_a: float, elo_b: float) -> float:
    """Expected score for team A vs team B (probability A wins)."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def compute_elo_per_game(manifest: pd.DataFrame | Path | None = None) -> pd.DataFrame:
    """
    Compute pre-game ELO for home and away for every game in the manifest.

    Process: group by season_id, sort by game_date. Within each season, each
    team starts at ELO_INITIAL. After each game, update both teams using
    standard ELO: new_elo = old_elo + K * (actual - expected); actual = 1 if
    win else 0.

    manifest: DataFrame with columns game_id, game_date, season_id,
              home_team_id, away_team_id, wl_home (W/L) or pts_home/pts_away.
              If Path, load from CSV (default: RAW_DIR / games_manifest.csv).

    Returns: DataFrame with columns game_id, home_elo, away_elo (pre-game).
    """
    if manifest is None:
        manifest = pd.read_csv(RAW_DIR / "games_manifest.csv", dtype={"game_id": str})
    elif isinstance(manifest, (Path, str)):
        manifest = pd.read_csv(manifest, dtype={"game_id": str})

    manifest = manifest.copy()
    manifest["game_id"] = manifest["game_id"].astype(str).str.zfill(10)
    if "game_date" not in manifest.columns:
        raise ValueError("manifest must have game_date")
    manifest["game_date"] = pd.to_datetime(manifest["game_date"], errors="coerce")
    manifest = manifest.dropna(subset=["game_date", "season_id", "home_team_id", "away_team_id"])

    # Winner: home wins = 1, away wins = 0
    if "wl_home" in manifest.columns:
        manifest["home_won"] = (manifest["wl_home"].astype(str).str.upper() == "W").astype(int)
    else:
        manifest["home_won"] = (manifest["pts_home"] > manifest["pts_away"]).astype(int)

    results = []
    for season_id, grp in manifest.groupby("season_id", sort=False):
        grp = grp.sort_values("game_date")
        # current ELO per team (team_id -> elo)
        elo = {}
        for _, row in grp.iterrows():
            gid = str(row["game_id"]).zfill(10)
            hid = int(row["home_team_id"])
            aid = int(row["away_team_id"])
            home_won = int(row["home_won"])

            home_elo = elo.get(hid, ELO_INITIAL)
            away_elo = elo.get(aid, ELO_INITIAL)
            results.append({"game_id": gid, "home_elo": home_elo, "away_elo": away_elo})

            expected_home = _expected_score(home_elo, away_elo)
            expected_away = _expected_score(away_elo, home_elo)
            elo[hid] = home_elo + ELO_K * (home_won - expected_home)
            elo[aid] = away_elo + ELO_K * ((1 - home_won) - expected_away)

    return pd.DataFrame(results)
