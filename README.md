# NBA Win Probability Engine

Predict in-game win probability from play-by-play: **fetch** → **parse** → **replay** (with per-season ELO) → **train** a logistic regression. One pipeline from raw NBA data to a saved model.

---

## What it does

| Step | Script | Output |
|------|--------|--------|
| **Fetch** | `run_pipeline.py` | Raw play-by-play CSVs + `games_manifest.csv` |
| **Parse** | `run_pipeline.py --parse-only` | `events_dataset.csv` + per-game `*_events.csv` |
| **Replay** | `run_replay.py` | `training_dataset.csv` (state features + ELO + label) |
| **Train** | `run_train.py` | `win_prob_model.joblib` (model + scaler) |

**Features:** time left, score differential, period, possession, home court, **home/away ELO** (per season).  
**Label:** did the home team win the game?

### Model performance (typical run, 80/20 game split)

| Metric | Value | Meaning |
|--------|--------|--------|
| **Validation accuracy** | ~76% | Fraction of events where the model’s predicted winner (home vs away) matches the actual game winner. |
| **Brier score** | ~0.16 | Mean squared error between predicted P(home win) and 0/1 outcome. Lower is better; 0.25 = no better than 50%; ~0.16 indicates probabilities are somewhat calibrated. |
| **Log loss** | ~0.48 | Proper scoring rule for probabilities; lower is better. |
| **First 8 min accuracy** | ~67% | Accuracy on events in the first 8 minutes only — where ELO helps most (score is still close). |

**Coefficients (what drives the model):** Score differential has the largest weight (the model mostly “follows the score”). Home/away ELO add team strength so early-game predictions aren’t 50/50. Time left and period have smaller effects. The model outputs **P(home wins)** at each moment; Brier and calibration reflect how reliable those probability numbers are.

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Fetch one season (or --all-seasons for 2022-23, 2023-24, 2024-25)
python run_pipeline.py --season 2023-24 --limit 100

# 3. Parse raw → events
python run_pipeline.py --parse-only

# 4. Replay games → training rows (with ELO from manifest)
python run_replay.py

# 5. Train and save model
python run_train.py
```

You’ll see validation accuracy and “first 8 min” accuracy (where ELO helps most), plus coefficients.

---

## Pipeline in detail

### Fetch

- **nba_api** `LeagueGameFinder` for game list; play-by-play from **cdn.nba.com** (no stats.nba.com timeouts).
- Saves `data/raw/<game_id>.csv` and merges into `data/raw/games_manifest.csv` (game_id, game_date, season_id, home/away team, final score).

```bash
python run_pipeline.py --season 2024-25          # one season
python run_pipeline.py --all-seasons             # 2022-23, 2023-24, 2024-25
python run_pipeline.py --season 2023-24 --limit 50
python run_pipeline.py --parse-only              # only parse existing raw files
```

### Parse

- Converts raw play-by-play into **normalized events** (period, time, score, possession, event_type, points_scored).
- Writes **`data/parsed/events_dataset.csv`** (all events + `label_home_win`, final scores) and **`data/parsed/games/<game_id>_events.csv`** per game.

### Replay + ELO

- **Replay:** For each game, start at tip-off and `apply_event(state, event)` at every play; emit (state features, label) for training.
- **ELO:** Per **season** (from manifest), teams start at 1500; after each game ELO updates by the usual formula (K=20). Each training row gets **home_elo** and **away_elo** (pre-game) so the model can use team strength, especially early in games.

```bash
python run_replay.py                  # all games, ELO from manifest
python run_replay.py --every-n 10     # sample every 10th event
python run_replay.py --no-elo        # skip ELO (no manifest needed)
```

### Train

- **Split:** 80% / 20% of **games** (not rows) for train/val so no game leaks across splits.
- **Scale:** `StandardScaler` on train only.
- **Model:** `LogisticRegression`; reports validation log loss, **Brier score**, accuracy, and accuracy in the **first 8 minutes** (where ELO matters most).

```bash
python run_train.py
```

### Frontend data (current season — not for training)

To run the **existing model** on 2025-26 games (e.g. for a frontend where you pick a game and see win prob play-by-play), fetch and parse into a separate directory so training data stays untouched:

```bash
python run_frontend_data.py           # fetch all 2025-26 games so far, parse + ELO
python run_frontend_data.py --limit 20   # test with 20 games
python run_frontend_data.py --parse-only # re-parse only (no fetch)
```

Output: **`data/frontend/raw/`** (manifest + raw CSVs), **`data/frontend/parsed/games/<id>_events.csv`**, **`data/frontend/parsed/game_elos.csv`**. Use **`src.inference.run_game_win_prob(game_id)`** to get play-by-play with `prob_home_win` for each event (no retraining).

### Web frontend

A small Flask app lets you search games by team or game ID and view win probability play-by-play:

```bash
# From project root (after run_frontend_data.py has populated data/frontend/)
python run_web.py
# or: python web/app.py
```

Open **http://127.0.0.1:5000**. Use **Team** to filter 2025-26 games, or enter a **Game ID** and click Load. Click a game to see a graph of win probability over every event; use the slider to scrub through and see the score and play description at each moment.

---

## Data layout

```
data/
├── raw/                        # Training seasons (2022-23, 2023-24, 2024-25)
│   ├── games_manifest.csv      # game_id, game_date, season_id, home/away team, pts, wl_home
│   └── <game_id>.csv           # raw play-by-play (one per game)
├── frontend/                   # Current season only (2025-26), not used for training
│   ├── raw/                    # games_manifest.csv + <game_id>.csv
│   └── parsed/
│       ├── games/<id>_events.csv
│       └── game_elos.csv       # game_id, home_elo, away_elo
└── parsed/
    ├── events_dataset.csv      # all events + label_home_win, pts_*_final
    ├── training_dataset.csv    # one row per (game, event): features + home_elo, away_elo, label_home_win
    ├── win_prob_model.joblib   # { "model", "scaler" } (after run_train.py)
    └── games/
        └── <game_id>_events.csv   # events for one game (for replay input)
```

**Training features:** `time_remaining_sec`, `score_diff`, `period`, `possession_home`, `is_home_court`, `home_elo`, `away_elo`.

---

## Project layout

```
nba_prob_engine/
├── data/raw/              # Raw play-by-play + manifest
├── data/parsed/           # Events, training CSV, model, per-game CSVs
├── docs/
├── src/
│   ├── config.py          # Paths, default season
│   ├── fetch.py           # get_games(), fetch_playbyplay(), fetch_all(_seasons)
│   ├── parse_events.py    # parse_pbp(), parse_all_games()
│   ├── game_state.py      # GameState, apply_event(), initial_state(), to_feature_dict()
│   ├── elo.py             # compute_elo_per_game() per season
│   ├── replay_games.py     # replay_one_game(), replay_all_games()
│   └── train.py           # load_training_data(), train_val_split(), scale, fit, evaluate, save
├── run_pipeline.py        # Fetch + parse CLI
├── run_replay.py          # Replay + ELO → training_dataset.csv
├── run_train.py           # Train model, print metrics, save joblib
├── requirements.txt
└── README.md
```

---

## Requirements

- **Python 3.10+**
- **nba_api**, **pandas**, **numpy**, **scikit-learn**, **requests** (see `requirements.txt`)

Play-by-play is fetched from **cdn.nba.com**; no API key needed.

---

## Using the trained model

**Play-by-play for a frontend game (2025-26):**

```python
from src.inference import run_game_win_prob

# Returns list of { event_num, period, time_remaining_sec, home_score, away_score,
#                   score_diff, possession, event_type, description, prob_home_win }
rows = run_game_win_prob("0022500001")  # game_id from data/frontend/parsed/games/
for r in rows[:5]:
    print(r["event_num"], r["home_score"], r["away_score"], r["prob_home_win"])
```

**Single-state prediction (e.g. custom state):** load `win_prob_model.joblib`, build a feature dict with `time_remaining_sec`, `score_diff`, `period`, `possession_home`, `is_home_court`, `home_elo`, `away_elo`, then `scaler.transform` and `model.predict_proba`.

---

## CLI reference

| Script | Option | Description |
|--------|--------|-------------|
| **run_pipeline** | `--season X` | Season (e.g. `2023-24`). |
| | `--all-seasons` | Fetch 2022-23, 2023-24, 2024-25. |
| | `--limit N` | Max games per season (testing). |
| | `--fetch-only` | Only fetch; don’t parse. |
| | `--parse-only` | Only parse existing raw CSVs. |
| **run_replay** | `--every-n N` | Emit a row every N events (default 1). |
| | `--limit N` | Replay only N games (testing). |
| | `--no-elo` | Don’t add ELO (no manifest). |
| **run_train** | *(none)* | Load data, split, scale, train, evaluate, save. |

---

## License

Use and modify as you like. Data from NBA.com; respect their terms for production use.
