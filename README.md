# FPL Optimizer

An advanced Fantasy Premier League optimizer that uses **expected points (xP) modeling** and linear programming to maximize gameweek points and recommend optimal transfers.

## Key Features

- **Advanced xP Calculator**: Uses xG, xA, BPS modeling, penalty tracking, and fixture analysis
- **Squad Optimization**: Build the optimal 15-player squad within budget constraints
- **Transfer Planning**: Multi-gameweek transfer recommendations with hit analysis
- **Captain Selection**: Data-driven captain picks based on expected points
- **Web UI**: React dashboard showing your squad, captain picks, and recommendations
- **CLI Tools**: Command-line interface for quick analysis

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or with Poetry
poetry install
```

### Configure FPL Authentication

Create a `.env` file:

```bash
FPL_TEAM_ID=your_team_id_here
```

**For OAuth login (Google/Facebook)**, extract cookies from your browser:

1. Log into FPL at https://fantasy.premierleague.com
2. Open DevTools (F12) → Application/Storage tab
3. Find cookies for `fantasy.premierleague.com`
4. Copy `pl_profile` and `sessionid` values to `.env`:

```bash
FPL_COOKIE_PL_PROFILE=your_pl_profile_cookie
FPL_COOKIE_SESSIONID=your_sessionid_cookie
```

**For email/password login**:

```bash
FPL_EMAIL=your_email@example.com
FPL_PASSWORD=your_password
```

### Usage

**Web UI (Recommended)**:

```bash
# Terminal 1: Start backend
python -m fpl_optimizer.server

# Terminal 2: Start frontend
cd frontend && npm run dev

# Open browser to http://localhost:5173
```

**CLI**:

```bash
# Optimize a new squad
python -m fpl_optimizer.main optimize

# Get transfer recommendations
python -m fpl_optimizer.main transfers

# Get captain recommendations
python -m fpl_optimizer.main captain

# Run with advanced xP (includes Understat scraping)
python -m fpl_optimizer.main optimize --use-advanced-xp
```

## Advanced xP Calculator

The optimizer uses a sophisticated expected points model that provides significantly better predictions than basic form-based approaches:

### What's Different?

| Feature | Basic Model | Advanced xP |
|---------|-------------|-------------|
| **Goals** | Form-based | xG + penalty modeling |
| **Assists** | Form-based | xA + set piece tracking |
| **Bonus** | Price guess | BPS formula modeling |
| **Clean Sheets** | FDR-based | Poisson probability |
| **Minutes** | Average | Rotation & congestion |
| **Form** | Static | Weighted (recent>old) |

### Features

- **xG/xA Integration**: Uses expected goals/assists from Understat
- **Penalty Modeling**: Tracks penalty takers (1st choice vs backup) with conversion rates
- **BPS-Based Bonus**: Models actual Bonus Point System formula (goals=24-42 BPS, assists=18, etc.)
- **Clean Sheet Probability**: Poisson distribution based on team strengths
- **Fixture Congestion**: Adjusts for rotation risk in dense schedules
- **Form Weighting**: 70% recent games, 30% season average
- **Risk Assessment**: Provides ceiling (90th %), floor (10th %), and variance
- **Differential Value**: Adjusts xP for ownership (for rank climbing)

### Example Output

```python
Mohamed Salah (£13.0M) - Liverpool
──────────────────────────────────────────
  Total xP:          7.84 points
  Expected Minutes:  89 mins
  Goal xP:          0.523 goals → 2.61 pts
  Assist xP:        0.312 assists → 0.94 pts
  Clean Sheet:      42.3% → 0.42 pts
  Bonus:            1.65 pts (BPS: 38.2)
  Ceiling (90%):    12.8 pts
  Floor (10%):      3.2 pts
```

### Using Advanced xP

```python
from fpl_optimizer.data_sources import build_advanced_stats
from fpl_optimizer.expected_points import AdvancedExpectedPointsCalculator

# Fetch advanced stats (takes ~30 seconds, scrapes Understat)
player_stats, team_strength = build_advanced_stats(
    players, teams, use_understat=True
)

# Create calculator
calculator = AdvancedExpectedPointsCalculator(player_stats, team_strength)

# Get detailed breakdown
breakdown = calculator.calculate(player, gameweek, fixtures, teams)
print(f"xP: {breakdown.total_expected_points:.2f}")
```

See `example_advanced_xp.py` for complete examples.

## Architecture

```
fpl-optimizer/
├── fpl_optimizer/
│   ├── fpl_client.py          # FPL API client with auth
│   ├── models.py              # Pydantic data models
│   ├── expected_points.py     # Advanced xP calculator
│   ├── data_sources.py        # Understat/FBref scrapers
│   ├── optimizer.py           # MILP optimization
│   ├── server.py              # FastAPI backend
│   └── main.py                # CLI interface
├── frontend/                  # React + TypeScript UI
├── test_advanced_xp.py        # Validation tests
├── example_advanced_xp.py     # Usage examples
└── .env                       # Your credentials
```

## How It Works

### 1. Data Collection
- **FPL API**: Official player stats, fixtures, teams
- **Understat** (optional): xG, xA, shot data
- **FBref** (optional): Advanced stats (tackles, key passes)

### 2. Expected Points Calculation
- **Simple Mode**: Form + fixture difficulty (fast)
- **Advanced Mode**: xG/xA + BPS + penalties + congestion (accurate)

### 3. Optimization
Mixed Integer Linear Programming with constraints:
- Budget: £100M
- Squad: 15 players (2 GKP, 5 DEF, 5 MID, 3 FWD)
- Formation: 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD
- Max 3 players per team
- Transfer planning with multi-gameweek lookahead

## Web UI Features

- **Squad View**: See your starting XI and bench
- **Captain Picks**: Top 5 recommendations with expected points
- **Player Cards**: Show position, cost, points, form
- **Chips Tracking**: View available chips
- **Real-time Data**: Fetches live FPL data

## CLI Commands

```bash
# Optimize new squad
python -m fpl_optimizer.main optimize [--objective points|value|differential]

# Get transfer recommendations
python -m fpl_optimizer.main transfers [--num-transfers 1]

# Captain recommendations
python -m fpl_optimizer.main captain

# Test advanced xP system
python test_advanced_xp.py
```

## FPL Constraints

The optimizer respects all official rules:
- **Budget**: £100M total (£1000 in tenths)
- **Squad**: 2 GKP, 5 DEF, 5 MID, 3 FWD
- **Starting XI**: 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD
- **Team Limit**: Max 3 players from same team
- **Transfers**: 1 free per GW, -4 points for additional

## Authentication Guide

### Getting Cookies (OAuth Login)

**Chrome/Edge**:
1. Log into FPL → F12 → Application tab
2. Expand Cookies → fantasy.premierleague.com
3. Copy `sessionid` and `pl_profile` values

**Firefox**:
1. Log into FPL → F12 → Storage tab
2. Cookies → fantasy.premierleague.com
3. Copy values

**Safari**:
1. Enable Develop menu → Show Web Inspector
2. Storage → Cookies
3. Copy values

Cookies typically last several weeks. When expired, just get fresh ones.

## Dependencies

Core:
- **pandas**: Data manipulation
- **requests**: HTTP client
- **pulp**: Linear programming
- **pydantic**: Data validation
- **beautifulsoup4**: Web scraping

Optional (advanced xP):
- **understat** data (scraped)
- **fbref** data (scraped)

## Development

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run tests
python test_advanced_xp.py

# Format code
black fpl_optimizer/

# Run examples
python example_advanced_xp.py
```

## Troubleshooting

**Authentication errors**:
- Check `.env` has correct credentials
- Get fresh cookies from browser
- Verify team ID is correct

**"Cannot fetch team"**:
- Make sure you're logged into FPL
- Check cookies haven't expired
- Verify team ID in URL: `https://fantasy.premierleague.com/entry/{team_id}/event/{gw}`

**Advanced xP slow**:
- First run scrapes Understat (~30 seconds)
- Use simple calculator for speed
- Cache player_stats/team_strength objects

**Optimization fails**:
- Check solver is installed (CBC)
- Verify player data loaded correctly
- Try simpler objective (points vs differential)

## Performance

| Operation | Simple xP | Advanced xP |
|-----------|-----------|-------------|
| Calculate single player | < 1ms | < 5ms |
| Calculate all players | ~0.5s | ~2s |
| Scrape Understat | N/A | ~30s |
| Full optimization | ~2s | ~5s |

## Roadmap

- [x] Advanced xP calculator with xG/xA
- [x] BPS-based bonus modeling
- [x] Penalty tracking
- [x] Fixture congestion adjustment
- [x] Web UI with React
- [ ] Historical backtesting
- [ ] Machine learning (XGBoost)
- [ ] Injury probability modeling
- [ ] Chip optimization (Wildcard, Free Hit, etc.)
- [ ] Discord bot integration

## License

MIT

## Contributing

Contributions welcome! See issues for ideas or submit PRs.

Key areas:
- Improve xP accuracy (better bonus model, set piece tracking)
- Add data sources (FotMob, FPL Review API)
- Backtest validation
- Machine learning models
- UI improvements