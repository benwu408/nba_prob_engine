# NBA Win Probability Engine

Data pipeline (Step 0) for the Sports Win Probability project: fetch historical NBA play-by-play from **nba_api**, parse it into normalized events, and produce a single dataset for training a win-probability model.

---

## Overview

1. **Fetch** — Get game list for a season (e.g. 2023–24) via `LeagueGameFinder`, then download raw play-by-play per game from the `PlayByPlay` endpoint. Saves one CSV per game under `data/raw/` and a `games_manifest.csv`.
2. **Parse** — Turn each raw play-by-play CSV into a sequence of **normalized events** (score, time, possession, event type). Merge all games into **`events_dataset.csv`** with labels for training (`label_home_win`, final scores).

Later steps (not in this repo yet): define `GameState`, `apply_event()`, train a model, and build the real-time + viz pipeline.

---

## What’s included

- **Code:** `src/fetch.py`, `src/parse_events.py`, `src/config.py`, and `run_pipeline.py` (CLI).
- **Sample data:** One **sample game** (`0022301234`) in `data/raw/` so you can run the parser without calling the NBA API. This is not a full season — only enough to validate the pipeline and output format.

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ (uses `|` for optional types).

---

## Quick start (no API)

Run the parser on the included sample game:

```bash
python run_pipeline.py --parse-only
```

You should see:

- **`data/parsed/events_dataset.csv`** — all events with labels.
- **`data/parsed/games/0022301234_events.csv`** — events for that one game (no labels).

---

## Full pipeline (fetch + parse)

Fetch games for a season and download play-by-play (then parse by default):

```bash
python run_pipeline.py --season 2023-24
```

To only download raw data and skip parsing:

```bash
python run_pipeline.py --season 2023-24 --fetch-only
```

To re-parse after you have more raw files (without re-fetching):

```bash
python run_pipeline.py --parse-only
```

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--season` | `2023-24` | Season (e.g. `2022-23`). |
| `--limit N` | None | Max number of games to fetch (for testing). |
| `--fetch-only` | off | Only fetch; do not parse. |
| `--parse-only` | off | Only parse existing raw CSVs; do not fetch. |
| `--raw-dir DIR` | `data/raw` | Directory for raw CSVs and manifest. |

---

## Data format

### Raw (per game)

- **`data/raw/<game_id>.csv`** — nba_api play-by-play columns: `GAME_ID`, `EVENTNUM`, `EVENTMSGTYPE`, `PERIOD`, `PCTIMESTRING`, `HOMEDESCRIPTION`, `VISITORDESCRIPTION`, `SCORE`, etc.
- **`data/raw/games_manifest.csv`** — one row per game: `game_id`, `game_date`, `home_team_id`, `away_team_id`, `home_team_abbrev`, `away_team_abbrev`, `pts_home`, `pts_away`, `wl_home`.

### Parsed (dataset)

**`data/parsed/events_dataset.csv`** — one row per event:

| Column | Description |
|--------|-------------|
| `game_id` | 10-character game ID. |
| `event_num` | Event number in the feed. |
| `period` | Quarter (1–4+, OT). |
| `time_remaining_sec` | Seconds left in the period. |
| `home_score` | Home team score after this event. |
| `away_score` | Away team score after this event. |
| `possession` | `home` or `away` when inferrable; else empty. |
| `event_type` | See below. |
| `points_scored` | Points on this play (0, 1, 2, or 3). |
| `description` | Short play description. |
| `label_home_win` | 1 if home won the game, 0 otherwise. |
| `pts_home_final` | Final home score. |
| `pts_away_final` | Final away score. |

**Event types:** `made_2`, `made_3`, `miss_2`, `miss_3`, `free_throw_made`, `free_throw_miss`, `turnover`, `rebound`, `foul`, `timeout`, `jump_ball`, `period_start`, `period_end`, `substitution`, `violation`, `ejection`, `instant_replay`, `other`.

**`data/parsed/games/<game_id>_events.csv`** — same event columns per game, without `label_home_win` / `pts_*_final`, for replay and Step 2.

---

## Project layout

```
nba_prob_engine/
├── data/
│   ├── raw/                      # Raw play-by-play: <game_id>.csv + games_manifest.csv
│   └── parsed/                   # events_dataset.csv + games/<game_id>_events.csv
├── src/
│   ├── config.py                 # Paths, DEFAULT_SEASON, REQUEST_DELAY_SEC
│   ├── fetch.py                  # get_games(), fetch_playbyplay(), fetch_all()
│   └── parse_events.py           # parse_pbp(), parse_all_games(), load_game_manifest()
├── run_pipeline.py               # CLI entrypoint
├── requirements.txt
└── README.md
```

---

## nba_api (library we use)

**[nba_api](https://github.com/swar/nba_api)** is a Python client for NBA.com’s APIs. It does not provide its own data; it calls stats.nba.com (and optionally live endpoints) and returns JSON or Pandas DataFrames.

- **Install:** `pip install nba_api` (we list it in `requirements.txt`).
- **Docs:** [GitHub repo](https://github.com/swar/nba_api), [Table of Contents](https://github.com/swar/nba_api/blob/master/docs/table_of_contents.md), [Endpoints](https://github.com/swar/nba_api/tree/master/docs/nba_api/stats/endpoints). [Stats Examples](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/examples.md) (proxy, custom headers, timeout) — local copy: [`docs/nba_api_stats_examples.md`](docs/nba_api_stats_examples.md). Jupyter: [Finding Games](https://github.com/swar/nba_api/blob/master/docs/examples/Finding%20Games.ipynb), [PlayByPlay](https://github.com/swar/nba_api/blob/master/docs/examples/PlayByPlay.ipynb).
- **What this project uses:**
  - **`nba_api.stats.endpoints.leaguegamefinder.LeagueGameFinder`** — list games by season (and optional team). We use `season_nullable`, `season_type_nullable=SeasonType.regular`. Returns one row per team per game (so two rows per game); we merge to one row per game with home/away.
  - **`nba_api.stats.endpoints.playbyplay.PlayByPlay`** — raw play-by-play for one game (param `game_id`). Returns a DataFrame with columns like `GAME_ID`, `EVENTNUM`, `EVENTMSGTYPE`, `PERIOD`, `PCTIMESTRING`, `HOMEDESCRIPTION`, `VISITORDESCRIPTION`, `SCORE`, `SCOREMARGIN`. We parse these in `src/parse_events.py`.
- **Other useful parts of nba_api:** `nba_api.stats.static.teams` / `players` for IDs and names; `nba_api.live.nba.endpoints` for live scoreboard/game data (we don’t use those in this pipeline).

---

## API notes (behavior / gotchas)

**Fetch uses cdn.nba.com (live) by default** so it does not block or timeout on stats.nba.com:

- We request `https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json` and convert the response into the same DataFrame shape as stats, so the parser is unchanged.
- **No stats.nba.com call** is made unless you set `try_stats_first=True` in code (e.g. if you want to try stats then fall back to live).

So `python run_pipeline.py --season 2023-24` should complete without timeouts. If a game has no data on cdn.nba.com (rare), that game will be in the failed list.

---

## Next steps (roadmap)

- **Step 1** — Define `GameState` (time, score, possession, period, home-court, etc.) and update it from parsed events.
- **Step 2** — Implement `parse_event(raw) -> Event` and `apply_event(state, event) -> new_state`.
- **Step 3** — Build training examples: sample (state → label) from replayed games.
- **Step 4+** — Train baseline model (e.g. logistic regression / GBDT), real-time loop, visualization.
