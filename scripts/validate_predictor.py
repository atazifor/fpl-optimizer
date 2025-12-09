"""
Validate predictor weights using time-series cross-validation.

Uses the historical data we already collected for the backtest to:
1. Test different weight combinations
2. Evaluate prediction accuracy vs actual points
3. Find optimal weights that minimize prediction error
"""

import json
import logging
import pickle
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Import PlayerGWData from backtest script
sys.path.insert(0, str(Path(__file__).parent))
from full_backtest import PlayerGWData

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PredictionConfig:
    """Configuration for prediction weights."""
    elite_threshold: float = 7.0  # pts/90 to be "elite"
    good_threshold: float = 5.0   # pts/90 to be "good"

    # Elite player weights (pts/90 >= 7.0)
    elite_season_weight: float = 0.70
    elite_form_weight: float = 0.30

    # Good player weights (5.0 <= pts/90 < 7.0)
    good_season_weight: float = 0.50
    good_form_weight: float = 0.50

    # Average player weights (pts/90 < 5.0)
    avg_season_weight: float = 0.40
    avg_form_weight: float = 0.60

    # Fixture difficulty multiplier scale
    fixture_weight: float = 0.05  # ±10% max


def calculate_prediction(
    total_points: int,
    minutes: int,
    form: float,
    config: PredictionConfig
) -> float:
    """
    Calculate predicted points using config weights.

    This mirrors the logic in predictor.py _calculate_form_points()
    """
    if minutes < 90:
        return 0.0

    # Season quality
    season_quality = (total_points / minutes) * 90

    # Games played
    games_played = minutes / 90

    # Reliability factor
    if games_played < 3:
        reliability = 0.5
    elif games_played < 5:
        reliability = 0.7
    elif games_played < 8:
        reliability = 0.85
    else:
        reliability = 1.0

    season_quality *= reliability

    # Blend based on quality tier
    if games_played >= 10 and season_quality > 0 and form > 0:
        if season_quality >= config.elite_threshold:
            blended = config.elite_season_weight * season_quality + config.elite_form_weight * form
        elif season_quality >= config.good_threshold:
            blended = config.good_season_weight * season_quality + config.good_form_weight * form
        else:
            blended = config.avg_season_weight * season_quality + config.avg_form_weight * form

        # Quality floor
        if season_quality >= 4.0:
            blended = max(blended, season_quality * 0.6)

        # Form spike protection
        if form > season_quality * 1.5:
            blended = min(blended, season_quality * 1.3)

        return blended
    elif season_quality > 0:
        return season_quality
    elif form > 0:
        return form

    return 0.0


def evaluate_config(
    historical_db: Dict,
    config: PredictionConfig,
    start_gw: int = 5,  # Start from GW5 (need history)
    end_gw: int = 14
) -> Tuple[float, float]:
    """
    Evaluate a configuration on historical data.

    Returns (MAE, RMSE) - lower is better
    """
    predictions = []
    actuals = []

    for gw in range(start_gw, end_gw + 1):
        if gw not in historical_db:
            continue

        for player_id, player_data in historical_db[gw].items():
            # Skip if no actual points
            if player_data.gw_points == 0 and player_data.gw_minutes == 0:
                continue

            # Make prediction
            pred = calculate_prediction(
                player_data.total_points - player_data.gw_points,  # Exclude current GW
                player_data.minutes - player_data.gw_minutes,
                player_data.form,
                config
            )

            predictions.append(pred)
            actuals.append(player_data.gw_points)

    # Calculate errors
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

    return mae, rmse


def grid_search_weights(historical_db: Dict) -> PredictionConfig:
    """
    Grid search over weight combinations to find optimal config.
    """
    logger.info("=" * 80)
    logger.info("GRID SEARCH FOR OPTIMAL WEIGHTS")
    logger.info("=" * 80)

    best_config = None
    best_mae = float('inf')

    # Test different weight combinations
    elite_weights = [0.60, 0.70, 0.80, 0.90]
    good_weights = [0.40, 0.50, 0.60, 0.70]
    avg_weights = [0.30, 0.40, 0.50, 0.60]

    total_combos = len(elite_weights) * len(good_weights) * len(avg_weights)
    tested = 0

    logger.info(f"Testing {total_combos} weight combinations...\n")

    for elite_w in elite_weights:
        for good_w in good_weights:
            for avg_w in avg_weights:
                config = PredictionConfig(
                    elite_season_weight=elite_w,
                    elite_form_weight=1.0 - elite_w,
                    good_season_weight=good_w,
                    good_form_weight=1.0 - good_w,
                    avg_season_weight=avg_w,
                    avg_form_weight=1.0 - avg_w,
                )

                mae, rmse = evaluate_config(historical_db, config)
                tested += 1

                if mae < best_mae:
                    best_mae = mae
                    best_config = config
                    logger.info(f"✅ New best! MAE={mae:.3f}, RMSE={rmse:.3f}")
                    logger.info(f"   Elite: {elite_w:.0%} season / {1-elite_w:.0%} form")
                    logger.info(f"   Good:  {good_w:.0%} season / {1-good_w:.0%} form")
                    logger.info(f"   Avg:   {avg_w:.0%} season / {1-avg_w:.0%} form\n")

    logger.info("=" * 80)
    logger.info(f"TESTED {tested} CONFIGURATIONS")
    logger.info("=" * 80)

    return best_config


def compare_with_baseline(historical_db: Dict):
    """
    Compare current weights vs optimal weights.
    """
    logger.info("\n" + "=" * 80)
    logger.info("COMPARING CURRENT VS OPTIMAL WEIGHTS")
    logger.info("=" * 80)

    # Current config (from predictor.py)
    current_config = PredictionConfig(
        elite_season_weight=0.70,
        elite_form_weight=0.30,
        good_season_weight=0.50,
        good_form_weight=0.50,
        avg_season_weight=0.40,
        avg_form_weight=0.60,
    )

    current_mae, current_rmse = evaluate_config(historical_db, current_config)

    logger.info("\nCURRENT WEIGHTS:")
    logger.info(f"  Elite: 70% season / 30% form")
    logger.info(f"  Good:  50% season / 50% form")
    logger.info(f"  Avg:   40% season / 60% form")
    logger.info(f"  MAE:  {current_mae:.3f}")
    logger.info(f"  RMSE: {current_rmse:.3f}")

    # Find optimal
    logger.info("\nSearching for optimal weights...\n")
    optimal_config = grid_search_weights(historical_db)

    optimal_mae, optimal_rmse = evaluate_config(historical_db, optimal_config)

    logger.info("\nOPTIMAL WEIGHTS:")
    logger.info(f"  Elite: {optimal_config.elite_season_weight:.0%} season / {optimal_config.elite_form_weight:.0%} form")
    logger.info(f"  Good:  {optimal_config.good_season_weight:.0%} season / {optimal_config.good_form_weight:.0%} form")
    logger.info(f"  Avg:   {optimal_config.avg_season_weight:.0%} season / {optimal_config.avg_form_weight:.0%} form")
    logger.info(f"  MAE:  {optimal_mae:.3f}")
    logger.info(f"  RMSE: {optimal_rmse:.3f}")

    improvement = (current_mae - optimal_mae) / current_mae * 100
    logger.info(f"\n{'IMPROVEMENT:' if improvement > 0 else 'DIFFERENCE:'} {improvement:+.1f}%")

    logger.info("=" * 80)


def main():
    """Run validation."""

    # Load historical database from backtest
    cache_file = Path("backtest_cache") / "historical_db_gw14.pkl"

    if not cache_file.exists():
        logger.error(f"❌ Historical database not found: {cache_file}")
        logger.error("   Run scripts/full_backtest.py first to build the database")
        return

    logger.info("Loading historical database...")
    with open(cache_file, 'rb') as f:
        historical_db = pickle.load(f)

    logger.info(f"✅ Loaded {len(historical_db)} gameweeks\n")

    # Run comparison
    compare_with_baseline(historical_db)


if __name__ == "__main__":
    main()