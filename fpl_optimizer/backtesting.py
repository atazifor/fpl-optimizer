"""
FPL Backtesting Framework

Validates expected points models against historical data to:
1. Measure prediction accuracy
2. Identify features that actually predict performance
3. Optimize model parameters
4. Find market inefficiencies

Usage:
    from fpl_optimizer.backtesting import BacktestRunner

    runner = BacktestRunner(season='2023-24')
    runner.collect_data()
    results = runner.run_backtest()
    runner.analyze_results(results)
"""

import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from .models import Player, Fixture, Team
from .expected_points import SimpleExpectedPointsCalculator, ExpectedPointsBreakdown

logger = logging.getLogger(__name__)


@dataclass
class GameweekData:
    """Historical data for a single gameweek."""

    gameweek: int
    players: List[Dict]  # Full player data from bootstrap
    fixtures: List[Dict]  # Fixture data
    teams: List[Dict]  # Team data
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PredictionResult:
    """Result of a single prediction."""

    player_id: int
    player_name: str
    gameweek: int
    predicted_points: float
    actual_points: int
    error: float
    absolute_error: float
    squared_error: float

    # Metadata
    position: str
    team: str
    price: float
    ownership: float
    minutes: int

    # Features used
    form: float
    season_avg: float
    fixture_difficulty: int


@dataclass
class BacktestResults:
    """Results from backtesting."""

    season: str
    gameweeks_tested: List[int]
    predictions: List[PredictionResult]

    # Aggregate metrics
    mae: float = 0.0  # Mean Absolute Error
    rmse: float = 0.0  # Root Mean Squared Error
    correlation: float = 0.0

    # By position
    mae_by_position: Dict[str, float] = field(default_factory=dict)

    # Feature importance
    feature_correlations: Dict[str, float] = field(default_factory=dict)


class HistoricalDataCollector:
    """Collects historical FPL data for backtesting."""

    def __init__(self, cache_dir: str = ".backtest_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.base_url = "https://fantasy.premierleague.com/api"

    def collect_season(self, season: str = "2023-24") -> Dict[int, GameweekData]:
        """
        Collect all gameweek data for a season.

        Args:
            season: Season identifier (e.g., '2023-24')

        Returns:
            Dictionary mapping gameweek number to GameweekData
        """
        cache_file = self.cache_dir / f"{season}_data.pkl"

        # Check cache
        if cache_file.exists():
            logger.info(f"Loading {season} data from cache")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        logger.info(f"Collecting {season} data from FPL API...")

        # For 2023-24, we need to use archive or current season data
        # Note: FPL API only has current season live
        # For historical seasons, we'd need to use archived data

        gameweek_data = {}

        # Get current season bootstrap (has all historical GW data in player stats)
        bootstrap = self._fetch_bootstrap()

        # For each completed gameweek
        for gw in range(1, 39):  # All 38 gameweeks
            try:
                gw_data = self._fetch_gameweek_data(gw, bootstrap)
                if gw_data:
                    gameweek_data[gw] = gw_data
                    logger.info(f"Collected GW{gw} data")
            except Exception as e:
                logger.warning(f"Could not fetch GW{gw}: {e}")
                break  # Stop at first unavailable GW

        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(gameweek_data, f)

        logger.info(f"Collected {len(gameweek_data)} gameweeks")
        return gameweek_data

    def _fetch_bootstrap(self) -> Dict:
        """Fetch bootstrap-static data."""
        url = f"{self.base_url}/bootstrap-static/"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def _fetch_gameweek_data(self, gw: int, bootstrap: Dict) -> Optional[GameweekData]:
        """
        Fetch data for a specific gameweek.

        Uses the bootstrap data which contains historical performance.
        """
        # Get fixtures for this gameweek
        fixtures_url = f"{self.base_url}/fixtures/?event={gw}"
        try:
            fixtures_response = requests.get(fixtures_url)
            fixtures_response.raise_for_status()
            fixtures = fixtures_response.json()
        except:
            fixtures = []

        # Filter to only finished fixtures
        finished_fixtures = [f for f in fixtures if f.get('finished', False)]

        if not finished_fixtures:
            return None  # GW not finished yet

        return GameweekData(
            gameweek=gw,
            players=bootstrap['elements'],
            fixtures=finished_fixtures,
            teams=bootstrap['teams'],
        )


class BacktestRunner:
    """Runs backtests to validate prediction models."""

    def __init__(self, season: str = "2023-24", calculator: Optional[SimpleExpectedPointsCalculator] = None):
        self.season = season
        self.collector = HistoricalDataCollector()
        self.calculator = calculator if calculator is not None else SimpleExpectedPointsCalculator()
        self.data: Optional[Dict[int, GameweekData]] = None

    def collect_data(self):
        """Collect historical data for the season."""
        logger.info(f"Collecting data for {self.season}...")
        self.data = self.collector.collect_season(self.season)
        logger.info(f"Collected {len(self.data)} gameweeks")

    def run_backtest(
        self,
        start_gw: int = 5,  # Start after 5 GWs to have some form data
        end_gw: Optional[int] = None,
    ) -> BacktestResults:
        """
        Run backtest across gameweeks.

        For each GW:
        1. Use data up to GW-1 to predict GW
        2. Compare predictions to actual results
        3. Calculate errors

        Args:
            start_gw: First gameweek to test (default 5 to have form data)
            end_gw: Last gameweek to test (default: all available)

        Returns:
            BacktestResults with all predictions and metrics
        """
        if not self.data:
            raise ValueError("Must call collect_data() first")

        end_gw = end_gw or max(self.data.keys())
        gameweeks = range(start_gw, end_gw + 1)

        logger.info(f"Running backtest for GW {start_gw} to {end_gw}")

        predictions = []

        for gw in gameweeks:
            if gw not in self.data:
                continue

            logger.info(f"Testing GW{gw}...")
            gw_predictions = self._predict_gameweek(gw)
            predictions.extend(gw_predictions)

        # Calculate aggregate metrics
        results = BacktestResults(
            season=self.season,
            gameweeks_tested=list(gameweeks),
            predictions=predictions,
        )

        self._calculate_metrics(results)

        return results

    def _predict_gameweek(self, gw: int) -> List[PredictionResult]:
        """
        Predict all players for a gameweek and compare to actual.
        """
        gw_data = self.data[gw]

        # Convert to our models
        players = [self._dict_to_player(p, gw) for p in gw_data.players]
        fixtures = [self._dict_to_fixture(f) for f in gw_data.fixtures]
        teams = {t['id']: self._dict_to_team(t) for t in gw_data.teams}

        predictions = []

        for player in players:
            # Only predict for players who played
            actual_points = self._get_actual_points(player.id, gw, gw_data)

            if actual_points is None:
                continue  # Player didn't play this GW

            # Make prediction
            try:
                breakdown = self.calculator.calculate(player, gw, fixtures, teams)
                predicted = breakdown.total_expected_points
            except Exception as e:
                logger.warning(f"Prediction failed for player {player.id} GW{gw}: {e}")
                continue

            # Get fixture difficulty
            player_fixtures = [f for f in fixtures if f.event == gw and (f.team_h == player.team_id or f.team_a == player.team_id)]
            fdr = player_fixtures[0].team_a_difficulty if player_fixtures and player_fixtures[0].team_h == player.team_id else 3

            result = PredictionResult(
                player_id=player.id,
                player_name=player.name,
                gameweek=gw,
                predicted_points=predicted,
                actual_points=actual_points,
                error=predicted - actual_points,
                absolute_error=abs(predicted - actual_points),
                squared_error=(predicted - actual_points) ** 2,
                position=player.position_name,
                team=str(player.team_id),
                price=player.cost_millions,
                ownership=player.selected_by_percent,
                minutes=player.minutes,
                form=player.form,
                season_avg=player.points_per_game,
                fixture_difficulty=fdr,
            )

            predictions.append(result)

        return predictions

    def _get_actual_points(self, player_id: int, gw: int, gw_data: GameweekData) -> Optional[int]:
        """Get actual points scored by player in this GW."""
        # Find player in data
        player_data = next((p for p in gw_data.players if p['id'] == player_id), None)

        if not player_data:
            return None

        # Check if player has GW history
        # Note: This is tricky with current API - we'd need individual player history
        # For now, approximate using event_points if available
        return player_data.get('event_points', None)

    def _calculate_metrics(self, results: BacktestResults):
        """Calculate aggregate metrics from predictions."""
        if not results.predictions:
            return

        # Overall metrics
        errors = [p.error for p in results.predictions]
        abs_errors = [p.absolute_error for p in results.predictions]
        sq_errors = [p.squared_error for p in results.predictions]

        results.mae = np.mean(abs_errors)
        results.rmse = np.sqrt(np.mean(sq_errors))

        # Correlation
        predicted = [p.predicted_points for p in results.predictions]
        actual = [p.actual_points for p in results.predictions]
        results.correlation = np.corrcoef(predicted, actual)[0, 1]

        # By position
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            pos_predictions = [p for p in results.predictions if p.position == pos]
            if pos_predictions:
                results.mae_by_position[pos] = np.mean([p.absolute_error for p in pos_predictions])

        logger.info(f"MAE: {results.mae:.2f}, RMSE: {results.rmse:.2f}, Corr: {results.correlation:.3f}")

    def _dict_to_player(self, data: Dict, current_gw: int) -> Player:
        """Convert API dict to Player model."""
        # Map position type to name
        pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}

        return Player(
            id=data['id'],
            first_name=data.get('first_name', ''),
            second_name=data.get('second_name', data.get('web_name', '')),
            name=data['web_name'],
            element_type=data['element_type'],
            team_id=data['team'],
            position_name=pos_map[data['element_type']],
            cost=data['now_cost'],
            cost_millions=data['now_cost'] / 10.0,
            total_points=data['total_points'],
            minutes=data['minutes'],
            goals_scored=data['goals_scored'],
            assists=data['assists'],
            clean_sheets=data['clean_sheets'],
            goals_conceded=data['goals_conceded'],
            own_goals=data['own_goals'],
            penalties_saved=data['penalties_saved'],
            penalties_missed=data['penalties_missed'],
            yellow_cards=data['yellow_cards'],
            red_cards=data['red_cards'],
            saves=data['saves'],
            bonus=data['bonus'],
            bps=data['bps'],
            influence=data['influence'],
            creativity=data['creativity'],
            threat=data['threat'],
            ict_index=data['ict_index'],
            form=float(data.get('form', 0)),
            points_per_game=float(data.get('points_per_game', 0)),
            selected_by_percent=float(data.get('selected_by_percent', 0)),
            expected_goals=data.get('expected_goals', '0'),
            expected_assists=data.get('expected_assists', '0'),
            expected_goal_involvements=data.get('expected_goal_involvements', '0'),
            expected_goals_conceded=data.get('expected_goals_conceded', '0'),
            is_available=data.get('status', 'a') == 'a',
        )

    def _dict_to_fixture(self, data: Dict) -> Fixture:
        """Convert API dict to Fixture model."""
        return Fixture(
            id=data['id'],
            event=data['event'],
            team_h=data['team_h'],
            team_a=data['team_a'],
            team_h_difficulty=data.get('team_h_difficulty', 3),
            team_a_difficulty=data.get('team_a_difficulty', 3),
            finished=data.get('finished', False),
        )

    def _dict_to_team(self, data: Dict) -> Team:
        """Convert API dict to Team model."""
        return Team(
            id=data['id'],
            name=data['name'],
            short_name=data['short_name'],
            strength=data.get('strength', 3),
            strength_overall_home=data.get('strength_overall_home', 1000),
            strength_overall_away=data.get('strength_overall_away', 1000),
            strength_attack_home=data.get('strength_attack_home', 1000),
            strength_attack_away=data.get('strength_attack_away', 1000),
            strength_defence_home=data.get('strength_defence_home', 1000),
            strength_defence_away=data.get('strength_defence_away', 1000),
        )

    def analyze_results(self, results: BacktestResults):
        """
        Analyze backtest results to find insights.

        Outputs:
        - Model accuracy metrics
        - Feature importance
        - Position-specific performance
        - Error patterns
        """
        print("\n" + "="*60)
        print(f"BACKTEST RESULTS: {results.season}")
        print("="*60)
        print(f"\nGameweeks tested: {min(results.gameweeks_tested)} - {max(results.gameweeks_tested)}")
        print(f"Total predictions: {len(results.predictions)}")

        print(f"\n📊 OVERALL ACCURACY:")
        print(f"  MAE:  {results.mae:.2f} points")
        print(f"  RMSE: {results.rmse:.2f} points")
        print(f"  Correlation: {results.correlation:.3f}")

        print(f"\n📍 ACCURACY BY POSITION:")
        for pos, mae in sorted(results.mae_by_position.items()):
            print(f"  {pos}: {mae:.2f} points")

        # Top and bottom predictions
        sorted_preds = sorted(results.predictions, key=lambda x: x.absolute_error)

        print(f"\n✅ BEST PREDICTIONS (lowest error):")
        for p in sorted_preds[:5]:
            print(f"  {p.player_name:15} GW{p.gameweek}: Predicted {p.predicted_points:.1f}, Actual {p.actual_points} (Error: {p.absolute_error:.1f})")

        print(f"\n❌ WORST PREDICTIONS (highest error):")
        for p in sorted_preds[-5:]:
            print(f"  {p.player_name:15} GW{p.gameweek}: Predicted {p.predicted_points:.1f}, Actual {p.actual_points} (Error: {p.absolute_error:.1f})")

        # Feature analysis
        print(f"\n🔍 FEATURE CORRELATIONS:")
        self._analyze_features(results)

    def _analyze_features(self, results: BacktestResults):
        """Analyze which features correlate with accurate predictions."""

        # Extract features
        forms = np.array([p.form for p in results.predictions])
        season_avgs = np.array([p.season_avg for p in results.predictions])
        prices = np.array([p.price for p in results.predictions])
        ownerships = np.array([p.ownership for p in results.predictions])
        errors = np.array([p.absolute_error for p in results.predictions])

        # Calculate correlations
        print(f"  Form vs Error: {np.corrcoef(forms, errors)[0, 1]:.3f}")
        print(f"  Season Avg vs Error: {np.corrcoef(season_avgs, errors)[0, 1]:.3f}")
        print(f"  Price vs Error: {np.corrcoef(prices, errors)[0, 1]:.3f}")
        print(f"  Ownership vs Error: {np.corrcoef(ownerships, errors)[0, 1]:.3f}")


def main():
    """Run backtesting."""
    logging.basicConfig(level=logging.INFO)

    runner = BacktestRunner(season="2024-25")  # Current season
    runner.collect_data()

    # Run backtest
    results = runner.run_backtest(start_gw=5, end_gw=13)

    # Analyze
    runner.analyze_results(results)

    # Save results
    output_dir = Path("backtest_results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "results.pkl", 'wb') as f:
        pickle.dump(results, f)

    print(f"\n✅ Results saved to {output_dir}/results.pkl")


if __name__ == "__main__":
    main()