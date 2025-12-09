#!/usr/bin/env python3
"""
Parallel Blend Ratio Backtesting

Runs 3 backtests in parallel with different predictive/reactive blend ratios:
- 90/10: Very predictive (trust season quality > form)
- 80/20: Balanced predictive
- 70/30: More reactive (original configuration)

Usage:
    python scripts/parallel_blend_backtest.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict
import json
import logging

from fpl_optimizer.expected_points import SimpleExpectedPointsCalculator
from fpl_optimizer.backtesting import BacktestRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BlendConfig:
    """Configuration for a blend ratio test."""
    elite_predictive: float
    elite_reactive: float
    good_predictive: float
    good_reactive: float
    avg_predictive: float
    avg_reactive: float
    name: str


# Test configurations
BLEND_CONFIGS = [
    BlendConfig(0.90, 0.10, 0.80, 0.20, 0.70, 0.30, "90/10 (Very Predictive)"),
    BlendConfig(0.80, 0.20, 0.70, 0.30, 0.60, 0.40, "80/20 (Balanced)"),
    BlendConfig(0.70, 0.30, 0.50, 0.50, 0.40, 0.60, "70/30 (More Reactive)"),
]


def run_single_backtest(config: BlendConfig) -> Dict:
    """
    Run a full backtest with a specific blend configuration.

    Args:
        config: BlendConfig with blend ratio weights

    Returns:
        Dict with backtest results
    """
    print(f"\n{'='*80}")
    print(f"Starting backtest: {config.name}")
    print(f"{'='*80}")
    print(f"Elite: {config.elite_predictive*100:.0f}/{config.elite_reactive*100:.0f}")
    print(f"Good:  {config.good_predictive*100:.0f}/{config.good_reactive*100:.0f}")
    print(f"Avg:   {config.avg_predictive*100:.0f}/{config.avg_reactive*100:.0f}")

    try:
        # Create calculator with custom blend ratios
        calculator = SimpleExpectedPointsCalculator(
            elite_predictive_weight=config.elite_predictive,
            elite_reactive_weight=config.elite_reactive,
            good_predictive_weight=config.good_predictive,
            good_reactive_weight=config.good_reactive,
            avg_predictive_weight=config.avg_predictive,
            avg_reactive_weight=config.avg_reactive,
        )

        # Create backtest runner with custom calculator
        runner = BacktestRunner(season="2024-25", calculator=calculator)

        # Collect data (cached after first run)
        runner.collect_data()

        # Run backtest from GW5 to current GW
        results = runner.run_backtest(start_gw=5)

        # Extract summary
        total_predictions = len(results.predictions)

        # Calculate "total points" as negative MAE (higher is better)
        # We use -MAE so that lower error = higher score
        score = -results.mae

        result = {
            'config': config.name,
            'blend_ratios': {
                'elite': f"{config.elite_predictive*100:.0f}/{config.elite_reactive*100:.0f}",
                'good': f"{config.good_predictive*100:.0f}/{config.good_reactive*100:.0f}",
                'avg': f"{config.avg_predictive*100:.0f}/{config.avg_reactive*100:.0f}",
            },
            'mae': results.mae,
            'rmse': results.rmse,
            'correlation': results.correlation,
            'total_predictions': total_predictions,
            'gameweeks': results.gameweeks_tested,
            'score': score,  # Higher is better
        }

        print(f"\n{config.name} Results:")
        print(f"  MAE:  {results.mae:.2f}")
        print(f"  RMSE: {results.rmse:.2f}")
        print(f"  Correlation: {results.correlation:.3f}")
        print(f"  Predictions: {total_predictions}")

        return result

    except Exception as e:
        print(f"\nERROR in {config.name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'config': config.name,
            'error': str(e),
            'mae': 999.0,
            'score': -999.0,
        }


def run_parallel_backtests():
    """Run all blend ratio tests in parallel."""
    print("\n" + "="*80)
    print("PARALLEL BLEND RATIO BACKTESTING")
    print("="*80)
    print("\nTesting 3 configurations to find optimal predictive/reactive balance:")
    print("  1. 90/10 - Very Predictive (trust season quality)")
    print("  2. 80/20 - Balanced")
    print("  3. 70/30 - More Reactive (original)")
    print("\nEach backtest validates the model against historical gameweek results.")
    print("Lower MAE (Mean Absolute Error) = Better predictions")
    print("\n" + "="*80)

    # Run in parallel
    print("\nRunning 3 backtests in parallel...")
    with mp.Pool(processes=3) as pool:
        results = pool.map(run_single_backtest, BLEND_CONFIGS)

    # Compare results
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    print(f"{'Configuration':<30} {'MAE':<10} {'RMSE':<10} {'Correlation':<12} {'Winner'}")
    print("-"*80)

    best_config = min(results, key=lambda r: r.get('mae', 999.0))
    for result in sorted(results, key=lambda r: r.get('mae', 999.0)):
        if 'error' in result:
            print(f"{result['config']:<30} {'ERROR':<10}")
        else:
            is_best = "BEST" if result['config'] == best_config['config'] else ""
            mae = result['mae']
            rmse = result['rmse']
            corr = result['correlation']
            print(f"{result['config']:<30} {mae:<10.2f} {rmse:<10.2f} {corr:<12.3f} {is_best}")

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    if 'error' not in best_config:
        print(f"\nBest configuration: {best_config['config']}")
        print(f"  MAE: {best_config['mae']:.2f} points")
        print(f"  RMSE: {best_config['rmse']:.2f} points")
        print(f"  Correlation: {best_config['correlation']:.3f}")
        print(f"\nBlend Ratios:")
        for tier, ratio in best_config['blend_ratios'].items():
            print(f"  {tier.capitalize()}: {ratio}")
    print("\n" + "="*80)

    return results


if __name__ == "__main__":
    print("\nPARALLEL BLEND RATIO BACKTEST")
    print("="*80)
    print("\nThis will run 3 full backtests in parallel to find the optimal")
    print("balance between predictive (season quality) and reactive (form) weighting.")
    print("\nEach backtest:")
    print("  - Uses historical data from GW5 onwards (2024-25 season)")
    print("  - Predicts expected points for each player/gameweek")
    print("  - Compares predictions to actual points scored")
    print("  - Calculates error metrics (MAE, RMSE, correlation)")
    print("\nNote: First run will cache historical data for faster subsequent runs.")
    print("="*80)

    choice = input("\nProceed with parallel backtests? (y/n): ")
    if choice.lower() == 'y':
        results = run_parallel_backtests()

        # Save results
        output_file = Path(__file__).parent.parent / "backtest_blend_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")
    else:
        print("\nBacktest cancelled.")