#!/usr/bin/env python3
"""
Compare Your Actual Performance vs Optimizer Recommendations

Fetches your actual FPL team history and compares to what each
blend ratio configuration would have recommended.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import requests
from dotenv import load_dotenv

# Load environment
load_dotenv()
TEAM_ID = os.getenv('FPL_TEAM_ID', '2947731')

def get_team_history(team_id: str):
    """Fetch gameweek-by-gameweek history for a team."""
    url = f"https://fantasy.premierleague.com/api/entry/{team_id}/history/"

    print(f"\nFetching history for team {team_id}...")
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    return data


def analyze_performance():
    """Analyze actual performance vs optimizer potential."""
    print("\n" + "="*80)
    print("YOUR ACTUAL PERFORMANCE vs OPTIMIZER POTENTIAL")
    print("="*80)

    # Get your actual history
    history = get_team_history(TEAM_ID)

    current = history.get('current', [])

    if not current:
        print("\nNo gameweek history found!")
        return

    # Calculate totals
    total_points = 0
    total_rank = 0
    gw_count = 0

    print(f"\n{'GW':<5} {'Points':<10} {'Overall Rank':<15} {'GW Rank':<15} {'Team Value'}")
    print("-"*80)

    for gw_data in current:
        gw = gw_data['event']
        points = gw_data['points']
        overall_rank = gw_data['overall_rank']
        gw_rank = gw_data['rank']
        value = gw_data['value'] / 10  # Convert to millions

        total_points += points
        gw_count += 1

        print(f"{gw:<5} {points:<10} {overall_rank:,<15} {gw_rank:,<15} £{value:.1f}M")

    avg_points = total_points / gw_count if gw_count > 0 else 0

    print("-"*80)
    print(f"\nSUMMARY (GW1 - GW{gw_count}):")
    print(f"  Total Points: {total_points}")
    print(f"  Average per GW: {avg_points:.1f}")
    print(f"  Current Overall Rank: {current[-1]['overall_rank']:,}")
    print(f"  Team Value: £{current[-1]['value']/10:.1f}M")

    # Load backtest results if available
    results_file = Path(__file__).parent.parent / "backtest_blend_results.json"

    if results_file.exists():
        import json
        with open(results_file) as f:
            backtest_results = json.load(f)

        print("\n" + "="*80)
        print("OPTIMIZER PREDICTION ACCURACY")
        print("="*80)
        print("\nNote: These measure prediction accuracy (MAE), not total team points.")
        print("Lower MAE = Better at predicting individual player performance")
        print()
        print(f"{'Configuration':<30} {'MAE':<10} {'RMSE':<10} {'Correlation'}")
        print("-"*80)

        for result in sorted(backtest_results, key=lambda r: r.get('mae', 999.0)):
            if 'error' not in result:
                mae = result['mae']
                rmse = result['rmse']
                corr = result['correlation']
                print(f"{result['config']:<30} {mae:<10.2f} {rmse:<10.2f} {corr:.3f}")

        print("\n" + "="*80)
        print("INTERPRETATION")
        print("="*80)
        print("\nThe backtest shows which blend ratio is best at PREDICTING player points.")
        print("This doesn't directly translate to your team's total score because:")
        print("  - Your team selection (which players you picked)")
        print("  - Your transfer decisions (when you bought/sold)")
        print("  - Your captain choices")
        print("  - Chips played (wildcards, bench boost, etc.)")
        print("\nTo see if the optimizer would have scored more points, we'd need to")
        print("simulate it actually managing a team from GW1 (like an AI manager).")

    else:
        print("\nBacktest results not found. Run parallel_blend_backtest.py first.")

    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        analyze_performance()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()