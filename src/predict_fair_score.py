"""Generate predictions and compute fair scorelines."""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Add parent directory to path to import fair_game
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fair_game.config import PROCESSED_DATA_DIR, MODELS_DIR

def main():
    # Load trained models
    home_model_file = f"{MODELS_DIR}/home_model.pkl"
    away_model_file = f"{MODELS_DIR}/away_model.pkl"

    if not os.path.exists(home_model_file) or not os.path.exists(away_model_file):
        print("ERROR: Models not found")
        print("Please run train_model.py first")
        sys.exit(1)

    print("Loading models...")
    home_model = joblib.load(home_model_file)
    away_model = joblib.load(away_model_file)
    print(f"Loaded: {home_model_file}")
    print(f"Loaded: {away_model_file}")

    # Read match dataset
    input_file = f"{PROCESSED_DATA_DIR}/match_dataset.csv"

    if not os.path.exists(input_file):
        print(f"ERROR: {input_file} not found")
        print("Please run build_dataset.py first")
        sys.exit(1)

    print(f"\nReading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} matches")

    # Prepare features for home model
    df['home_indicator'] = 1
    home_features = ['home_shots_total', 'home_shots_on_target', 'home_possession', 'home_indicator']
    X_home = df[home_features].fillna(0)

    # Prepare features for away model
    away_features = ['away_shots_total', 'away_shots_on_target', 'away_possession']
    X_away = df[away_features].fillna(0)

    # Generate predictions
    print("\nGenerating predictions...")
    predicted_home_goals = home_model.predict(X_home)
    predicted_away_goals = away_model.predict(X_away)

    # Compute fair scorelines (round and clip to [0, 6])
    print("Computing fair scorelines...")
    fair_home_goals = np.clip(np.round(predicted_home_goals), 0, 6).astype(int)
    fair_away_goals = np.clip(np.round(predicted_away_goals), 0, 6).astype(int)

    # Create output dataframe
    output_df = pd.DataFrame({
        'fixture_id': df['fixture_id'],
        'date': df['date'],
        'home_team_name': df['home_team_name'],
        'away_team_name': df['away_team_name'],
        'actual_home_goals': df['home_goals'],
        'actual_away_goals': df['away_goals'],
        'predicted_home_goals': predicted_home_goals,
        'predicted_away_goals': predicted_away_goals,
        'fair_home_goals': fair_home_goals,
        'fair_away_goals': fair_away_goals,
    })

    # Add some derived columns for analysis
    output_df['actual_score'] = output_df['actual_home_goals'].astype(str) + '-' + output_df['actual_away_goals'].astype(str)
    output_df['fair_score'] = output_df['fair_home_goals'].astype(str) + '-' + output_df['fair_away_goals'].astype(str)
    output_df['actual_goal_diff'] = output_df['actual_home_goals'] - output_df['actual_away_goals']
    output_df['fair_goal_diff'] = output_df['fair_home_goals'] - output_df['fair_away_goals']
    output_df['goal_diff_error'] = output_df['fair_goal_diff'] - output_df['actual_goal_diff']

    # Save to CSV
    output_file = f"{PROCESSED_DATA_DIR}/fair_scores.csv"
    output_df.to_csv(output_file, index=False)
    print(f"\nSaved predictions to {output_file}")
    print(f"Dataset shape: {output_df.shape}")

    # Show summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Matches processed: {len(output_df)}")
    print(f"\nActual goals:")
    print(f"  Home: {output_df['actual_home_goals'].mean():.2f} ± {output_df['actual_home_goals'].std():.2f}")
    print(f"  Away: {output_df['actual_away_goals'].mean():.2f} ± {output_df['actual_away_goals'].std():.2f}")
    print(f"\nPredicted goals (before rounding):")
    print(f"  Home: {output_df['predicted_home_goals'].mean():.2f} ± {output_df['predicted_home_goals'].std():.2f}")
    print(f"  Away: {output_df['predicted_away_goals'].mean():.2f} ± {output_df['predicted_away_goals'].std():.2f}")
    print(f"\nFair scoreline (after rounding/clipping):")
    print(f"  Home: {output_df['fair_home_goals'].mean():.2f} ± {output_df['fair_home_goals'].std():.2f}")
    print(f"  Away: {output_df['fair_away_goals'].mean():.2f} ± {output_df['fair_away_goals'].std():.2f}")

    # Show some example predictions
    print(f"\n=== Sample Predictions ===")
    sample = output_df[['home_team_name', 'away_team_name', 'actual_score', 'fair_score', 'goal_diff_error']].head(10)
    print(sample.to_string(index=False))

    # Show biggest mismatches
    print(f"\n=== Biggest Mismatches (by goal difference error) ===")
    output_df['abs_goal_diff_error'] = output_df['goal_diff_error'].abs()
    mismatches = output_df.nlargest(5, 'abs_goal_diff_error')[
        ['home_team_name', 'away_team_name', 'actual_score', 'fair_score', 'goal_diff_error']
    ]
    print(mismatches.to_string(index=False))

    # Calculate team-level luck
    print(f"\n=== Team Luck Analysis ===")

    # Create rows for each team's home matches
    home_luck = output_df[['home_team_name', 'actual_goal_diff', 'fair_goal_diff']].copy()
    home_luck.columns = ['team', 'actual_diff', 'fair_diff']

    # Create rows for each team's away matches (flip the sign)
    away_luck = output_df[['away_team_name', 'actual_goal_diff', 'fair_goal_diff']].copy()
    away_luck.columns = ['team', 'actual_diff', 'fair_diff']
    away_luck['actual_diff'] = -away_luck['actual_diff']
    away_luck['fair_diff'] = -away_luck['fair_diff']

    # Combine home and away
    team_luck = pd.concat([home_luck, away_luck], ignore_index=True)

    # Calculate luck: positive = got better results than deserved (lucky)
    team_luck['luck'] = team_luck['actual_diff'] - team_luck['fair_diff']

    # Aggregate by team
    team_stats = team_luck.groupby('team').agg({
        'luck': ['mean', 'sum', 'count']
    }).round(2)
    team_stats.columns = ['avg_luck', 'total_luck', 'matches']
    team_stats = team_stats.reset_index()
    team_stats = team_stats.sort_values('avg_luck', ascending=False)

    # Show luckiest teams
    print(f"\nLuckiest Teams (actual results better than fair results):")
    luckiest = team_stats.head(5)
    print(luckiest.to_string(index=False))

    # Show unluckiest teams
    print(f"\nUnluckiest Teams (actual results worse than fair results):")
    unluckiest = team_stats.tail(5).sort_values('avg_luck')
    print(unluckiest.to_string(index=False))

    # Save team luck analysis
    team_luck_file = f"{PROCESSED_DATA_DIR}/team_luck.csv"
    team_stats.to_csv(team_luck_file, index=False)
    print(f"\nSaved team luck analysis to {team_luck_file}")

if __name__ == '__main__':
    main()
