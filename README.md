# fair-game

Estimate "fair" scorelines from match statistics and compare them to actual results.

Fair-game analyzes soccer matches by training simple linear regression models on performance stats (shots, possession, shots on target) to predict what the score "should have been" based on underlying statistics, then compares these predictions to actual results to identify lucky and unlucky teams.

---

## What it does

1. Fetches match fixtures and team statistics from API-Football
2. Processes raw data into clean datasets
3. Trains two linear regression models (home goals, away goals)
4. Predicts "fair" scorelines based on match stats
5. Calculates team luck (which teams won matches they shouldn't have)
6. Generates visualizations and report

---

## Features

- Simple linear pipeline: 6 scripts, run in order
- Fair scoreline algorithm: round predictions, clip to [0,6]
- Team luck analysis: count matches where actual result differed from fair result
- Resume capability: API fetcher skips already-downloaded fixtures
- Rate limit handling: batched requests (10/minute) with progress tracking
- Interpretable models: linear regression with visible coefficients

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

Run the pipeline in order:

```bash
# 1. Verify API credentials
python src/smoke_test.py

# 2. Fetch data (~38 min for 380 matches at 10 req/min, because of the free plan limits)
python src/fetch_api.py

# 3. Build match-level dataset
python src/build_dataset.py

# 4. Train models
python src/train_model.py

# 5. Generate fair scorelines and team luck
python src/predict_fair_score.py

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

## Configuration

### Change league or season

Edit `fair_game/config.py`:

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

After changing, re-run from step 2:

```bash
python src/fetch_api.py
python src/build_dataset.py
python src/train_model.py
python src/predict_fair_score.py
python src/make_report.py
```

### Adjust rate limits

If you have a paid plan, edit `src/fetch_api.py`:

```python
BATCH_SIZE = 10     # Requests per batch
WAIT_TIME = 60      # Seconds between batches
```

---

## Project structure

```
model-drift-analysis/
├── src/
│   ├── smoke_test.py            # Verify API credentials
│   ├── fetch_api.py             # Download fixtures and stats
│   ├── build_dataset.py         # Transform to match-level data
│   ├── train_model.py           # Train prediction models
│   ├── predict_fair_score.py    # Generate fair scorelines
│   └── make_report.py           # Create visualizations
├── fair_game/
│   ├── config.py                # Configuration constants
│   └── __init__.py
├── data/                         # git-ignored
│   ├── raw/                     # Raw API JSON
│   └── processed/               # Cleaned CSVs
├── models/                       # git-ignored
│   ├── home_model.pkl
│   └── away_model.pkl
├── reports/                      # figures/ git-ignored
│   ├── report.md
│   └── figures/
├── .env                          # git-ignored
└── requirements.txt
```

---

## Data pipeline

```
API-Football
    ↓
data/raw/*.json
    ↓
data/processed/team_match_stats.csv (one row per team per match)
    ↓
data/processed/match_dataset.csv (one row per match)
    ↓
models/home_model.pkl, models/away_model.pkl
    ↓
data/processed/fair_scores.csv
    ↓
reports/report.md + reports/figures/*.png
```

---

## Model details

Two separate LinearRegression models:

**Home model:**
- Features: home_shots_total, home_shots_on_target, home_possession, home_indicator
- Target: home_goals

**Away model:**
- Features: away_shots_total, away_shots_on_target, away_possession
- Target: away_goals

**Fair scoreline:**
```python
fair_home_goals = clip(round(predicted_home_goals), 0, 6)
fair_away_goals = clip(round(predicted_away_goals), 0, 6)
```

**Team luck:**
- Lucky win: actual W, fair D/L
- Unlucky loss: actual L, fair D/W
- Net lucky results: (lucky_wins + lucky_draws) - (unlucky_losses + unlucky_draws)
- Luck rate: net lucky results as % of total matches

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

---

## License

MIT
