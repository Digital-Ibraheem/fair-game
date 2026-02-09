"""Configuration constants for fair-game project."""

import os

# API-Football constants (can be overridden by environment variables)
LEAGUE_ID = int(os.getenv('FAIR_GAME_LEAGUE_ID', 39))   # Default: Premier League
SEASON = int(os.getenv('FAIR_GAME_SEASON', 2023))       # Default: 2023 season
BASE_URL = "https://v3.football.api-sports.io"

# File paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
FIGURES_DIR = "reports/figures"

# Model configuration
DEFAULT_MODEL_TYPE = "linear"  # Options: "linear", "poisson"

# Feature sets for training
FEATURE_SETS = {
    "basic": ["shots_total", "shots_on_target", "possession"],
    "extended": [
        "shots_total", "shots_on_target", "possession",
        "shots_insidebox", "blocked_shots", "goalkeeper_saves",
        "pass_accuracy", "shot_efficiency"
    ],
    "xg": ["expected_goals"]
}
