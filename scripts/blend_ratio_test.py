#!/usr/bin/env python3
"""
Blend Ratio Parameter Sweep

Tests different predictive/reactive blend ratios to find optimal balance.

Compares:
- 90/10: Very predictive (trust season quality > form)
- 80/20: Balanced predictive
- 70/30: Original (more reactive to form)

Usage:
    python scripts/blend_ratio_test.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List

from fpl_optimizer.fpl_client import FPLClient
from fpl_optimizer.models import Player, Team, Fixture
from fpl_optimizer.expected_points import SimpleExpectedPointsCalculator

@dataclass
class BlendConfig:
    """Configuration for a blend ratio test."""
    elite_predictive: float  # Weight for baseline (predictive)
    elite_reactive: float    # Weight for form (reactive)
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


def monkey_patch_blend_ratio(config: BlendConfig):
    """Temporarily modify the blend ratios in the calculator."""
    import fpl_optimizer.expected_points as ep_module

    # Store original method
    original_calculate_baseline = ep_module.SimpleExpectedPointsCalculator._calculate_baseline

    def patched_calculate_baseline(self, player, captain_mode=False):
        """Patched version with custom blend ratios."""
        # Call original to get all the logic
        result = original_calculate_baseline(self, player, captain_mode)

        # The original method already does the blending, but we need to re-blend
        # This is a hack - ideally we'd pass blend ratios as parameters
        # For now, we'll just accept this approximation
        return result

    # This won't actually work as intended because the blending happens inside
    # the original function. We need a different approach.
    # Let's instead modify the source code inline for each test.
    pass


def run_backtest_with_config(config: BlendConfig) -> Dict:
    """
    Run backtest with specific blend configuration.

    This would need to:
    1. Modify expected_points.py blend ratios
    2. Run full backtest from GW1
    3. Return results

    For now, this is a placeholder showing the structure.
    """
    print(f"\n{'='*80}")
    print(f"Testing: {config.name}")
    print(f"{'='*80}")
    print(f"Elite players: {config.elite_predictive*100:.0f}% season quality, {config.elite_reactive*100:.0f}% form")
    print(f"Good players:  {config.good_predictive*100:.0f}% season quality, {config.good_reactive*100:.0f}% form")
    print(f"Avg players:   {config.avg_predictive*100:.0f}% season quality, {config.avg_reactive*100:.0f}% form")

    # TODO: Run actual backtest
    # For now, return mock results
    return {
        'config': config,
        'total_points': 0,
        'gameweeks': []
    }


def compare_blend_ratios():
    """Run all blend ratio tests and compare results."""
    print("\n" + "="*80)
    print("BLEND RATIO PARAMETER SWEEP")
    print("="*80)
    print("\nTesting optimal balance between predictive (season quality) and reactive (form)")
    print("\nNote: This requires modifying code for each test.")
    print("      A proper implementation would make blend ratios configurable parameters.")
    print("\n" + "="*80)

    # Sequential tests (parallel would require code modification infrastructure)
    results = []
    for config in BLEND_CONFIGS:
        result = run_backtest_with_config(config)
        results.append(result)

    # Compare results
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    print(f"{'Configuration':<30} {'Total Points':<15} {'Winner'}")
    print("-"*80)

    best_score = max(r['total_points'] for r in results)
    for result in results:
        is_best = "⭐ BEST" if result['total_points'] == best_score else ""
        print(f"{result['config'].name:<30} {result['total_points']:<15} {is_best}")

    return results


if __name__ == "__main__":
    print("\n⚠️  IMPORTANT: Blend Ratio Testing")
    print("="*80)
    print("This script shows the FRAMEWORK for testing different blend ratios.")
    print("\nTo actually run these tests, we need to:")
    print("1. Make blend ratios configurable parameters (not hardcoded)")
    print("2. Run full backtest for each configuration")
    print("3. Compare results")
    print("\nCurrently, the blend ratios are hardcoded in expected_points.py:1411-1419")
    print("="*80)

    choice = input("\nDo you want to see the framework anyway? (y/n): ")
    if choice.lower() == 'y':
        compare_blend_ratios()