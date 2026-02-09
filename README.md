# fair-game

Estimate "fair" scorelines from match statistics and compare them to actual results.

Fair-game analyzes soccer matches by training regression models on performance stats (shots, possession, xG) to predict what the score "should have been" based on underlying statistics, then compares these predictions to actual results to identify lucky and unlucky teams.

---

## What it does

1. Fetches match fixtures and team statistics from API-Football
2. Processes raw data into clean datasets with derived features
3. Trains regression models (Linear or Poisson) with cross-validation
4. Predicts "fair" scorelines based on match stats
5. Calculates team luck (which teams won matches they shouldn't have)
6. Generates visualizations and report

---

## Features

- **Unified pipeline runner**: Single command to run all steps
- **Multiple model types**: Linear regression or Poisson regression
- **Feature sets**: Basic (shots, possession), Extended (+ shot quality), or xG-based
- **Train/test split**: Honest evaluation with holdout set
- **Cross-validation**: K-fold CV for robust performance estimates
- **CLI arguments**: Configure league, season, and model without editing code
- **xG support**: Uses expected goals data when available
- **Resume capability**: API fetcher skips already-downloaded fixtures
- **Rate limit handling**: Batched requests (10/minute) with progress tracking

---

## Quick start

### Prerequisites

- Python 3.7+
- API key from [API-Football](https://www.api-football.com/) (free tier available)

### Installation

```bash
git clone <your-repo-url>
cd model-drift-analysis

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Setup

Create `.env` in project root:

```
FOOTBALL_API_KEY=your_api_key_here
FOOTBALL_API_HOST=v3.football.api-sports.io
```

Free tier: 100 requests/day. Get key at [api-football.com](https://www.api-football.com/).

---

## Usage

### Run the full pipeline

```bash
# Run everything with defaults
python src/run_pipeline.py

# Skip API fetch (use existing data)
python src/run_pipeline.py --skip-fetch

# Use Poisson model with extended features
python src/run_pipeline.py --skip-fetch --model poisson --features extended

# Fetch data for a different league
python src/run_pipeline.py --league 140 --season 2024  # La Liga
```

### Run individual steps

```bash
# 1. Verify API credentials
python src/smoke_test.py

# 2. Fetch data (~38 min for 380 matches at 10 req/min)
python src/fetch_api.py --league 39 --season 2023

# 3. Build match-level dataset
python src/build_dataset.py

# 4. Train models
python src/train_model.py --model poisson --features extended

# 5. Generate fair scorelines
python src/predict_fair_score.py --model poisson

# 6. Create visualizations and report
python src/make_report.py

# View report
open reports/report.md
```

### Output

Terminal shows team luck analysis:
```
Luckiest Teams (won matches they should have lost/drawn):
           team  matches  lucky_wins  unlucky_losses  net_lucky_results  luck_rate
    Arsenal         12           4               1                  3       25.0
```

Report includes:
- Actual vs predicted scatter plots
- Goal difference error histogram
- Team luck bar chart
- Top 20 biggest mismatches
- Team luck rankings

---

## CLI Options

### fetch_api.py
| Option | Description |
|--------|-------------|
| `--league ID` | League ID (default: 39 Premier League) |
| `--season YEAR` | Season year (default: 2023) |

### train_model.py
| Option | Description |
|--------|-------------|
| `--model TYPE` | `linear` or `poisson` (default: linear) |
| `--features SET` | `basic`, `extended`, or `xg` (default: basic) |
| `--cv-folds N` | Cross-validation folds (default: 5) |
| `--test-size F` | Test set proportion (default: 0.2) |
| `--no-split` | Train on all data (legacy mode) |

### predict_fair_score.py
| Option | Description |
|--------|-------------|
| `--model TYPE` | `linear` or `poisson` (default: linear) |

### run_pipeline.py
| Option | Description |
|--------|-------------|
| `--skip-fetch` | Skip API fetch, use existing data |
| `--skip-smoke` | Skip smoke test |
| `--league ID` | League ID to fetch |
| `--season YEAR` | Season year to fetch |
| `--model TYPE` | Model type for training |
| `--features SET` | Feature set for training |
| `--no-split` | Train on all data |

---

## Configuration

### Change league or season

Use CLI arguments (recommended):
```bash
python src/fetch_api.py --league 140 --season 2024
```

Or set environment variables:
```bash
export FAIR_GAME_LEAGUE_ID=140
export FAIR_GAME_SEASON=2024
```

Or edit `fair_game/config.py`:
```python
LEAGUE_ID = 39          # 39 = Premier League
SEASON = 2023           # 2023 season

# Other leagues:
# 140 = La Liga
# 78  = Bundesliga
# 135 = Serie A
# 61  = Ligue 1
```

Find league IDs: https://www.api-football.com/documentation-v3#tag/Leagues

---

## Project structure

```
model-drift-analysis/
├── src/
│   ├── run_pipeline.py          # Unified pipeline runner
│   ├── smoke_test.py            # Verify API credentials
│   ├── fetch_api.py             # Download fixtures and stats
│   ├── build_dataset.py         # Transform to match-level data
│   ├── train_model.py           # Train prediction models
│   ├── predict_fair_score.py    # Generate fair scorelines
│   ├── make_report.py           # Create visualizations
│   └── utils.py                 # Shared utility functions
├── fair_game/
│   ├── config.py                # Configuration constants
│   └── __init__.py
├── data/                         # git-ignored
│   ├── raw/                     # Raw API JSON
│   └── processed/               # Cleaned CSVs
├── models/                       # git-ignored
│   ├── home_model.pkl
│   ├── away_model.pkl
│   └── model_metadata.pkl       # Feature info for predictions
├── reports/                      # figures/ git-ignored
│   ├── report.md
│   └── figures/
├── .env                          # git-ignored
└── requirements.txt
```

---

## Model details

### Model types

**Linear Regression** (default):
- Simple, interpretable coefficients
- Works well for most cases

**Poisson Regression** (`--model poisson`):
- Better suited for count data (goals)
- Often more accurate predictions

### Feature sets

**Basic** (`--features basic`):
- shots_total, shots_on_target, possession

**Extended** (`--features extended`):
- Basic features plus:
- shots_insidebox, blocked_shots, goalkeeper_saves
- shot_efficiency, pass_accuracy

**xG** (`--features xg`):
- Uses expected goals directly (when available)

### Fair scoreline calculation
```python
fair_home_goals = clip(round(predicted_home_goals), 0, 6)
fair_away_goals = clip(round(predicted_away_goals), 0, 6)
```

### Team luck metrics
- **Lucky win**: actual W, fair D/L
- **Unlucky loss**: actual L, fair D/W
- **Net lucky results**: (lucky_wins + lucky_draws) - (unlucky_losses + unlucky_draws)
- **Luck rate**: net lucky results as % of total matches

---

## Troubleshooting

### Rate limit errors

```
ERROR for fixture 1035046: {'rateLimit': 'Too many requests...'}
```

The script handles this with batching. If interrupted, re-run `fetch_api.py` - it resumes automatically.

### Out of requests

Free tier = 100 requests/day. Full season = ~380 requests.

Solutions:
- Wait 24 hours for reset
- Upgrade to paid plan
- Use smaller league/cup

### Missing files

```
ERROR: data/processed/match_dataset.csv not found
```

Run the pipeline in order. Each step requires the previous step's output.

### Missing features warning

```
Warning: Missing home features: ['home_xg']
```

This happens when using `--features extended` or `--features xg` with data that was fetched before xG extraction was added. Re-run `build_dataset.py` to extract new features from existing JSON files.

---

## License

MIT
