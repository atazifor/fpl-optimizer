"""
Machine Learning Expected Points Predictor

Trains ML models on historical data to predict player performance.
Uses ensemble of models optimized for different positions.

Models:
- Gradient Boosting: Captures non-linear interactions
- Random Forest: Robust to outliers
- Ridge Regression: Interpretable baseline

Features:
- Form (last 3, 5 games)
- Season stats (points/90, goals/90, assists/90)
- Fixture difficulty
- Team strength
- Position
- Price
- Ownership
- Minutes trend
- Expected stats (xG, xA)
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

from .backtesting import BacktestRunner, BacktestResults, PredictionResult
from .models import Player, Fixture, Team
from .expected_points import ExpectedPointsCalculator, ExpectedPointsBreakdown

logger = logging.getLogger(__name__)


class MLExpectedPointsPredictor(ExpectedPointsCalculator):
    """
    ML-based expected points predictor.

    Uses ensemble of models trained on historical data.
    """

    def __init__(self, model_dir: str = "ml_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        # Models (one per position for better accuracy)
        self.models: Dict[str, GradientBoostingRegressor] = {}
        self.scalers: Dict[str, StandardScaler] = {}

        # Feature names for consistency
        self.feature_names = [
            'form',
            'points_per_game',
            'goals_per_90',
            'assists_per_90',
            'minutes',
            'xg_per_90',
            'xa_per_90',
            'fixture_difficulty',
            'team_strength_attack',
            'team_strength_defence',
            'is_home',
            'price',
            'ownership',
            'minutes_last_3',
            'points_last_3',
            'is_gkp',
            'is_def',
            'is_mid',
            'is_fwd',
        ]

        self.trained = False

    def train(self, backtest_results: BacktestResults):
        """
        Train ML models on historical backtest data.

        Args:
            backtest_results: Results from backtesting with actual outcomes
        """
        logger.info("Training ML models on historical data...")

        # Prepare training data
        X, y, positions = self._prepare_training_data(backtest_results)

        logger.info(f"Training on {len(X)} samples")

        # Train separate models per position
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            logger.info(f"Training {pos} model...")

            # Filter to position
            pos_mask = np.array([p == pos for p in positions])
            X_pos = X[pos_mask]
            y_pos = y[pos_mask]

            if len(X_pos) < 50:
                logger.warning(f"Not enough data for {pos} ({len(X_pos)} samples), skipping")
                continue

            # Scale features
            scaler = StandardScaler()
            X_pos_scaled = scaler.fit_transform(X_pos)

            # Train ensemble with stronger regularization to prevent overfitting
            model = GradientBoostingRegressor(
                n_estimators=100,  # Reduced from 200
                learning_rate=0.01,  # Reduced from 0.05
                max_depth=3,  # Reduced from 5 for less complexity
                min_samples_split=50,  # Increased from 20
                min_samples_leaf=25,  # Increased from 10
                subsample=0.7,  # Reduced from 0.8
                max_features='sqrt',  # Add feature subsampling
                random_state=42,
            )

            model.fit(X_pos_scaled, y_pos)

            # Evaluate
            cv_scores = cross_val_score(
                model, X_pos_scaled, y_pos,
                cv=min(5, len(X_pos) // 100),
                scoring='neg_mean_absolute_error'
            )
            mae = -cv_scores.mean()

            logger.info(f"{pos} Model MAE: {mae:.2f} points (CV)")

            # Save
            self.models[pos] = model
            self.scalers[pos] = scaler

        self.trained = True
        logger.info("✅ Training complete!")

    def _prepare_training_data(
        self,
        results: BacktestResults
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Convert backtest predictions to ML training data.

        Returns:
            (X, y, positions) where:
            - X: Feature matrix (n_samples, n_features)
            - y: Target vector (actual points)
            - positions: Position for each sample
        """
        features = []
        targets = []
        positions = []

        for pred in results.predictions:
            # Extract features
            feature_vec = self._extract_features_from_prediction(pred)
            features.append(feature_vec)
            targets.append(pred.actual_points)
            positions.append(pred.position)

        return np.array(features), np.array(targets), positions

    def _extract_features_from_prediction(self, pred: PredictionResult) -> List[float]:
        """Extract feature vector from a prediction result."""
        # Calculate derived features
        goals_per_90 = 0.0  # Would need from data
        assists_per_90 = 0.0
        xg_per_90 = 0.0
        xa_per_90 = 0.0
        minutes_last_3 = pred.minutes  # Approximate
        points_last_3 = pred.form * 3  # Approximate

        # Position one-hot encoding
        is_gkp = 1.0 if pred.position == 'GKP' else 0.0
        is_def = 1.0 if pred.position == 'DEF' else 0.0
        is_mid = 1.0 if pred.position == 'MID' else 0.0
        is_fwd = 1.0 if pred.position == 'FWD' else 0.0

        # Team strength (approximate from price)
        team_strength_attack = pred.price * 100  # Placeholder
        team_strength_defence = 1000  # Placeholder

        # Is home (assume neutral)
        is_home = 0.5

        return [
            pred.form,
            pred.season_avg,
            goals_per_90,
            assists_per_90,
            pred.minutes / 90,  # Games played
            xg_per_90,
            xa_per_90,
            pred.fixture_difficulty,
            team_strength_attack,
            team_strength_defence,
            is_home,
            pred.price,
            pred.ownership,
            minutes_last_3,
            points_last_3,
            is_gkp,
            is_def,
            is_mid,
            is_fwd,
        ]

    def calculate(
        self,
        player: Player,
        gameweek: int,
        fixtures: List[Fixture],
        teams: Dict[int, Team],
    ) -> ExpectedPointsBreakdown:
        """
        Predict expected points using ML model.

        Falls back to simple calculator if not trained.
        """
        if not self.trained:
            logger.warning("ML model not trained, using simple fallback")
            from .expected_points import SimpleExpectedPointsCalculator
            calc = SimpleExpectedPointsCalculator()
            return calc.calculate(player, gameweek, fixtures, teams)

        # Extract features for this player
        features = self._extract_features(player, gameweek, fixtures, teams)

        # Get model for position
        model = self.models.get(player.position_name)
        scaler = self.scalers.get(player.position_name)

        if model is None or scaler is None:
            logger.warning(f"No model for {player.position_name}, using fallback")
            from .expected_points import SimpleExpectedPointsCalculator
            calc = SimpleExpectedPointsCalculator()
            return calc.calculate(player, gameweek, fixtures, teams)

        # Scale and predict
        features_scaled = scaler.transform([features])
        predicted_points = model.predict(features_scaled)[0]

        # Sanity check: Cap predictions to reasonable ranges by position
        # These are realistic upper bounds based on historical FPL data
        position_caps = {
            'GKP': 8.0,   # Keepers rarely score >8
            'DEF': 10.0,  # Defenders max ~10 (clean sheet + attacking returns + bonus)
            'MID': 15.0,  # Midfielders can haul big
            'FWD': 15.0,  # Forwards can haul big
        }
        max_reasonable = position_caps.get(player.position_name, 12.0)

        # If prediction is unrealistic, blend with simple calculator
        if predicted_points > max_reasonable:
            logger.warning(
                f"ML prediction for {player.name} ({predicted_points:.1f}) exceeds cap "
                f"({max_reasonable}), blending with simple calculator"
            )
            from .expected_points import SimpleExpectedPointsCalculator
            calc = SimpleExpectedPointsCalculator()
            simple_pred = calc.calculate(player, gameweek, fixtures, teams)
            # Blend: 70% simple, 30% capped ML
            predicted_points = 0.7 * simple_pred.total_expected_points + 0.3 * max_reasonable

        # Create breakdown
        breakdown = ExpectedPointsBreakdown(
            player_id=player.id,
            gameweek=gameweek,
            total_expected_points=round(max(0, predicted_points), 2),
            confidence=0.6,  # Moderate confidence with limited training data
            variance=4.0,
            ownership=player.selected_by_percent,
        )

        # Set ceiling and floor
        breakdown.ceiling = round(min(predicted_points + 6, max_reasonable * 1.5), 2)
        breakdown.floor = round(max(0, predicted_points - 3), 2)

        return breakdown

    def _extract_features(
        self,
        player: Player,
        gameweek: int,
        fixtures: List[Fixture],
        teams: Dict[int, Team],
    ) -> List[float]:
        """Extract features for a player for prediction."""
        # Get player's fixtures
        player_fixtures = [
            f for f in fixtures
            if f.event == gameweek and (f.team_h == player.team_id or f.team_a == player.team_id)
        ]

        # Fixture difficulty
        if player_fixtures:
            fixture = player_fixtures[0]
            is_home = fixture.team_h == player.team_id
            fdr = fixture.team_a_difficulty if is_home else fixture.team_h_difficulty

            # Get opponent team
            opp_id = fixture.team_a if is_home else fixture.team_h
            opp_team = teams.get(opp_id)
        else:
            is_home = 0.5
            fdr = 3
            opp_team = None

        # Calculate derived stats
        games_played = player.minutes / 90 if player.minutes > 0 else 0
        goals_per_90 = (player.goals_scored / player.minutes * 90) if player.minutes > 0 else 0
        assists_per_90 = (player.assists / player.minutes * 90) if player.minutes > 0 else 0

        # xG/xA per 90
        try:
            xg = float(player.expected_goals) if player.expected_goals else 0.0
            xa = float(player.expected_assists) if player.expected_assists else 0.0
            xg_per_90 = (xg / player.minutes * 90) if player.minutes > 0 else 0
            xa_per_90 = (xa / player.minutes * 90) if player.minutes > 0 else 0
        except:
            xg_per_90 = 0
            xa_per_90 = 0

        # Team strength
        team = teams.get(player.team_id)
        if team:
            if is_home:
                team_str_attack = team.strength_attack_home
                team_str_defence = team.strength_defence_home
            else:
                team_str_attack = team.strength_attack_away
                team_str_defence = team.strength_defence_away
        else:
            team_str_attack = 1000
            team_str_defence = 1000

        # Recent form (approximate)
        minutes_last_3 = player.minutes  # Would need game-by-game data
        points_last_3 = player.form * 3  # Approximate

        # Position encoding
        is_gkp = 1.0 if player.position_name == 'GKP' else 0.0
        is_def = 1.0 if player.position_name == 'DEF' else 0.0
        is_mid = 1.0 if player.position_name == 'MID' else 0.0
        is_fwd = 1.0 if player.position_name == 'FWD' else 0.0

        return [
            player.form,
            player.points_per_game,
            goals_per_90,
            assists_per_90,
            games_played,
            xg_per_90,
            xa_per_90,
            fdr,
            team_str_attack,
            team_str_defence,
            is_home,
            player.cost_millions,
            player.selected_by_percent,
            minutes_last_3,
            points_last_3,
            is_gkp,
            is_def,
            is_mid,
            is_fwd,
        ]

    def save(self, filename: str = "ml_model.pkl"):
        """Save trained model to disk."""
        if not self.trained:
            raise ValueError("Model not trained yet")

        path = self.model_dir / filename
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers,
                'feature_names': self.feature_names,
            }, f)

        logger.info(f"✅ Model saved to {path}")

    def load(self, filename: str = "ml_model.pkl"):
        """Load trained model from disk."""
        path = self.model_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.models = data['models']
        self.scalers = data['scalers']
        self.feature_names = data['feature_names']
        self.trained = True

        logger.info(f"✅ Model loaded from {path}")


def train_and_evaluate():
    """Train ML model and compare to baseline."""
    import logging
    logging.basicConfig(level=logging.INFO)

    # Run backtest to get training data
    logger.info("Running backtest to collect training data...")
    runner = BacktestRunner(season="2024-25")
    runner.collect_data()
    results = runner.run_backtest(start_gw=5, end_gw=13)

    logger.info(f"Baseline MAE: {results.mae:.2f} points")
    logger.info(f"Baseline Correlation: {results.correlation:.3f}")

    # Train ML model
    logger.info("\nTraining ML model...")
    ml_predictor = MLExpectedPointsPredictor()
    ml_predictor.train(results)

    # Save model
    ml_predictor.save()

    logger.info("\n✅ ML model trained and saved!")
    logger.info("To use it, update optimizer.py to use MLExpectedPointsPredictor")


if __name__ == "__main__":
    train_and_evaluate()