"""Train two linear regression models to predict home and away goals."""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Add parent directory to path to import fair_game
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fair_game.config import PROCESSED_DATA_DIR, MODELS_DIR

def main():
    # Read match dataset
    input_file = f"{PROCESSED_DATA_DIR}/match_dataset.csv"

    if not os.path.exists(input_file):
        print(f"ERROR: {input_file} not found")
        print("Please run build_dataset.py first")
        sys.exit(1)

    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} matches")

    # Prepare features for home model
    print("\n=== Training Home Goals Model ===")
    home_features = ['home_shots_total', 'home_shots_on_target', 'home_possession']

    # Add home indicator (constant 1 for home advantage)
    df['home_indicator'] = 1
    home_features.append('home_indicator')

    X_home = df[home_features].fillna(0)
    y_home = df['home_goals']

    print(f"Features: {home_features}")
    print(f"Training samples: {len(X_home)}")
    print(f"Feature means: {X_home.mean().to_dict()}")

    # Train home model
    home_model = LinearRegression()
    home_model.fit(X_home, y_home)

    # Evaluate home model
    y_home_pred = home_model.predict(X_home)
    home_mae = mean_absolute_error(y_home, y_home_pred)
    home_rmse = np.sqrt(mean_squared_error(y_home, y_home_pred))
    home_r2 = r2_score(y_home, y_home_pred)

    print(f"\nHome Model Performance:")
    print(f"  MAE:  {home_mae:.3f}")
    print(f"  RMSE: {home_rmse:.3f}")
    print(f"  R²:   {home_r2:.3f}")
    print(f"\nCoefficients:")
    for feature, coef in zip(home_features, home_model.coef_):
        print(f"  {feature:25s}: {coef:8.4f}")
    print(f"  {'intercept':25s}: {home_model.intercept_:8.4f}")

    # Prepare features for away model
    print("\n=== Training Away Goals Model ===")
    away_features = ['away_shots_total', 'away_shots_on_target', 'away_possession']

    X_away = df[away_features].fillna(0)
    y_away = df['away_goals']

    print(f"Features: {away_features}")
    print(f"Training samples: {len(X_away)}")
    print(f"Feature means: {X_away.mean().to_dict()}")

    # Train away model
    away_model = LinearRegression()
    away_model.fit(X_away, y_away)

    # Evaluate away model
    y_away_pred = away_model.predict(X_away)
    away_mae = mean_absolute_error(y_away, y_away_pred)
    away_rmse = np.sqrt(mean_squared_error(y_away, y_away_pred))
    away_r2 = r2_score(y_away, y_away_pred)

    print(f"\nAway Model Performance:")
    print(f"  MAE:  {away_mae:.3f}")
    print(f"  RMSE: {away_rmse:.3f}")
    print(f"  R²:   {away_r2:.3f}")
    print(f"\nCoefficients:")
    for feature, coef in zip(away_features, away_model.coef_):
        print(f"  {feature:25s}: {coef:8.4f}")
    print(f"  {'intercept':25s}: {away_model.intercept_:8.4f}")

    # Save models
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

    home_model_file = f"{MODELS_DIR}/home_model.pkl"
    away_model_file = f"{MODELS_DIR}/away_model.pkl"

    joblib.dump(home_model, home_model_file)
    joblib.dump(away_model, away_model_file)

    print(f"\n=== Models Saved ===")
    print(f"Home model: {home_model_file}")
    print(f"Away model: {away_model_file}")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Trained on {len(df)} matches")
    print(f"Home goals - MAE: {home_mae:.3f}, R²: {home_r2:.3f}")
    print(f"Away goals - MAE: {away_mae:.3f}, R²: {away_r2:.3f}")

if __name__ == '__main__':
    main()
