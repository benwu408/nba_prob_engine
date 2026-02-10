"""
Training the win-probability model.

- load_training_data(): load training_dataset.csv into a pandas DataFrame.
- FEATURE_COLUMNS / TARGET_COLUMN: which columns are X vs y.
- get_X_y(): split a DataFrame into feature matrix X and target y.
- train_val_split(): split by game_id so no game appears in both train and val.
- scale_train_val(): fit StandardScaler on train, transform train and val; return scaled arrays and scaler.
- fit_model(): fit LogisticRegression on (X_train, y_train).
- evaluate(): compute validation log loss and accuracy.
- save_model(): save model + scaler to disk (joblib).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

from .config import PARSED_DIR

# Default path for the training CSV (from run_replay.py)
TRAINING_CSV = "training_dataset.csv"
# Default path for saved model bundle (model + scaler)
DEFAULT_MODEL_PATH = PARSED_DIR / "win_prob_model.joblib"

# Features (inputs) and target (what we predict).
# home_elo / away_elo are added when replay is run with ELO (from manifest).
FEATURE_COLUMNS = [
    "time_remaining_sec",
    "score_diff",
    "period",
    "possession_home",
    "is_home_court",
    "home_elo",
    "away_elo",
]
TARGET_COLUMN = "label_home_win"


def load_training_data(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Load the training dataset from CSV into a single DataFrame.

    csv_path: path to CSV (default: PARSED_DIR / training_dataset.csv).
    Returns: DataFrame with columns game_id, event_num, label_home_win, and feature columns.
    """
    path = Path(csv_path or PARSED_DIR / TRAINING_CSV)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}. Run run_replay.py first.")
    return pd.read_csv(path)


def get_X_y(df: pd.DataFrame):
    """
    Split DataFrame into feature matrix X and target y.

    Uses FEATURE_COLUMNS and TARGET_COLUMN. Only includes feature columns
    that exist in df (so training data without ELO still works). Returns (X, y).
    """
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not cols:
        raise ValueError(f"None of {FEATURE_COLUMNS} found in DataFrame columns: {list(df.columns)}")
    X = df[cols].copy()
    y = df[TARGET_COLUMN]
    return X, y


def train_val_split(
    df: pd.DataFrame,
    val_frac: float = 0.2,
    random_state: int = 42,
):
    """
    Split data by game_id so no game appears in both train and val.

    val_frac: fraction of games to use for validation (default 0.2 = 20%).
    random_state: seed for shuffling game ids (reproducible splits).

    Returns: (X_train, X_val, y_train, y_val).
    """
    rng = np.random.default_rng(random_state)
    game_ids = df["game_id"].unique()
    rng.shuffle(game_ids)
    n_val = max(1, int(len(game_ids) * val_frac))
    val_game_ids = set(game_ids[:n_val])
    train_game_ids = set(game_ids[n_val:])

    train_mask = df["game_id"].isin(train_game_ids)
    val_mask = df["game_id"].isin(val_game_ids)
    train_df = df.loc[train_mask]
    val_df = df.loc[val_mask]

    X_train, y_train = get_X_y(train_df)
    X_val, y_val = get_X_y(val_df)
    return X_train, X_val, y_train, y_val


def scale_train_val(X_train: pd.DataFrame, X_val: pd.DataFrame):
    """
    Fit StandardScaler on X_train and transform both train and val.

    Val is scaled using train's mean and std (no val info leaks). Returns
    (X_train_scaled, X_val_scaled, scaler). Scaler should be saved with the
    model so predictions use the same scaling.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled, scaler


def fit_model(X_train, y_train, random_state: int = 42):
    """Fit LogisticRegression on (X_train, y_train). Returns fitted model."""
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_val, y_val) -> dict:
    """Compute validation log loss and accuracy. Returns dict with log_loss, accuracy."""
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    return {
        "log_loss": log_loss(y_val, y_pred_proba),
        "accuracy": accuracy_score(y_val, y_pred),
    }


def save_model(model, scaler, path: Path | None = None):
    """Save model and scaler to a single joblib file. Load with joblib.load() for (model, scaler) or a dict."""
    path = Path(path or DEFAULT_MODEL_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, path)
