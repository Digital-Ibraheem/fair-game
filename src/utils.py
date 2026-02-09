"""Shared utility functions for fair-game pipeline."""

import os
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv


def get_api_key():
    """Load and validate API key from environment.

    Returns:
        tuple: (api_key, api_host)

    Exits:
        If API key is not set or has placeholder value.
    """
    load_dotenv()
    api_key = os.getenv('FOOTBALL_API_KEY')
    api_host = os.getenv('FOOTBALL_API_HOST', 'v3.football.api-sports.io')

    if not api_key or api_key == 'put_your_key_here':
        print("ERROR: FOOTBALL_API_KEY not set in .env file")
        print("Please add your API key to .env:")
        print("  FOOTBALL_API_KEY=your_key_here")
        sys.exit(1)

    return api_key, api_host


def check_file_exists(filepath, dependency_script=None):
    """Check if required input file exists, exit with helpful message if not.

    Args:
        filepath: Path to check
        dependency_script: Name of script that produces this file (for error message)

    Returns:
        The filepath if it exists

    Exits:
        If file does not exist.
    """
    if not os.path.exists(filepath):
        print(f"ERROR: {filepath} not found")
        if dependency_script:
            print(f"Please run {dependency_script} first")
        sys.exit(1)
    return filepath


def load_dataframe(filepath, dependency_script=None):
    """Load CSV file into DataFrame with existence check.

    Args:
        filepath: Path to CSV file
        dependency_script: Name of script that produces this file (for error message)

    Returns:
        pandas.DataFrame

    Exits:
        If file does not exist.
    """
    check_file_exists(filepath, dependency_script)
    return pd.read_csv(filepath)


def ensure_dir(dirpath):
    """Create directory if it doesn't exist.

    Args:
        dirpath: Directory path to create

    Returns:
        The dirpath (for chaining)
    """
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    return dirpath


def clean_numeric(value, default=0):
    """Clean a numeric value, returning default if invalid.

    Args:
        value: Value to clean (may be string, number, or None)
        default: Value to return if cleaning fails

    Returns:
        Cleaned numeric value or default
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value
    try:
        # Handle percentage strings like "55%"
        if isinstance(value, str) and value.endswith('%'):
            return float(value.rstrip('%'))
        return float(value)
    except (ValueError, TypeError):
        return default


def clean_possession(value):
    """Clean possession value (percentage string to float).

    Args:
        value: Possession value like "55%" or 55

    Returns:
        Float value (0-100 scale) or 0 if invalid
    """
    return clean_numeric(value, default=0)
