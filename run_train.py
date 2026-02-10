#!/usr/bin/env python3
"""
Full training pipeline: load data (2), define X/y (3), split by game (4),
scale (5), train (6), evaluate (7), save model (8).
"""
from src.train import (
    DEFAULT_MODEL_PATH,
    load_training_data,
    fit_model,
    evaluate,
    save_model,
    scale_train_val,
    train_val_split,
)


def main():
    print("Loading training data...")
    df = load_training_data()
    print(f"Loaded {len(df):,} rows, {df['game_id'].nunique():,} games")

    print("\nSplitting by game (80% train, 20% val)...")
    X_train, X_val, y_train, y_val = train_val_split(df, val_frac=0.2, random_state=42)
    feature_names = list(X_train.columns)
    # Early game = first 8 min (time_remaining_sec >= 2400) — ELO matters most here
    early_val_mask = (X_val["time_remaining_sec"].values >= 2400)
    n_games = df["game_id"].nunique()
    n_val_games = max(1, int(n_games * 0.2))
    n_train_games = n_games - n_val_games
    print(f"Train: {len(X_train):,} rows ({n_train_games:,} games)")
    print(f"Val:   {len(X_val):,} rows ({n_val_games:,} games)")

    print("\nScaling features (StandardScaler on train only)...")
    X_train, X_val, scaler = scale_train_val(X_train, X_val)
    print("Done.")

    print("\nTraining LogisticRegression...")
    model = fit_model(X_train, y_train, random_state=42)
    print("Done.")

    print("\nModel coefficients (positive = more likely home wins):")
    for name, coef in zip(feature_names, model.coef_[0]):
        print(f"  {name}: {coef:+.4f}")

    print("\nValidation metrics (all moments):")
    metrics = evaluate(model, X_val, y_val)
    print(f"  Log loss:  {metrics['log_loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    if early_val_mask.any():
        n_early = early_val_mask.sum()
        metrics_early = evaluate(model, X_val[early_val_mask], y_val.values[early_val_mask])
        print(f"\nValidation metrics (first 8 min only, {n_early:,} rows) — where ELO helps most:")
        print(f"  Accuracy: {metrics_early['accuracy']:.4f}")

    print("\nSaving model and scaler...")
    save_model(model, scaler, path=DEFAULT_MODEL_PATH)
    print(f"  Saved to {DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()
