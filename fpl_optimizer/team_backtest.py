"""
Team Performance Backtest

Compares your actual FPL performance vs what the optimizer would have recommended.

This simulates running the optimizer week-by-week from GW1, making the same decisions
it would have made, and comparing the results.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


@dataclass
class GameweekComparison:
    """Comparison of actual vs optimizer performance for a gameweek."""

    gameweek: int

    # Your actual performance
    actual_points: int
    actual_transfers: int
    actual_transfer_cost: int
    actual_captain_points: Optional[int] = None
    actual_bench_points: int = 0

    # Optimizer recommendations (simulated)
    optimizer_points: Optional[int] = None
    optimizer_transfers: Optional[int] = None
    optimizer_transfer_cost: Optional[int] = None
    optimizer_captain_points: Optional[int] = None

    # Difference
    points_diff: Optional[int] = None  # Positive = optimizer better

    # Notes
    notes: str = ""


@dataclass
class BacktestSummary:
    """Summary of backtest results."""

    gameweeks: List[GameweekComparison]

    # Totals
    actual_total_points: int = 0
    optimizer_total_points: int = 0
    points_difference: int = 0

    # By gameweek
    gameweeks_optimizer_better: int = 0
    gameweeks_actual_better: int = 0
    gameweeks_equal: int = 0

    # Transfer efficiency
    actual_total_transfer_cost: int = 0
    optimizer_total_transfer_cost: int = 0


class TeamBacktester:
    """
    Backtests optimizer performance vs your actual team.

    Note: This is a simplified backtest that compares outcomes, not decisions.
    A full backtest would need to simulate the optimizer running week-by-week
    with perfect hindsight removal.
    """

    def __init__(self, team_id: int, season: str = "2024-25"):
        self.team_id = team_id
        self.season = season
        self.base_url = "https://fantasy.premierleague.com/api"

    def fetch_team_history(self) -> Dict:
        """Fetch team's historical performance."""
        url = f"{self.base_url}/entry/{self.team_id}/history/"
        logger.info(f"Fetching team history from {url}")

        response = requests.get(url)
        response.raise_for_status()

        return response.json()

    def fetch_gameweek_picks(self, gameweek: int) -> Dict:
        """Fetch team's picks for a specific gameweek."""
        url = f"{self.base_url}/entry/{self.team_id}/event/{gameweek}/picks/"

        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Could not fetch GW{gameweek} picks: {e}")
            return {}

    def run_backtest(self, start_gw: int = 1, end_gw: Optional[int] = None) -> BacktestSummary:
        """
        Run backtest comparing actual vs optimizer performance.

        Note: This is a RETROSPECTIVE comparison showing what happened.
        It does NOT simulate live optimizer decisions (which would require
        removing hindsight bias and running the optimizer with only past data).

        Args:
            start_gw: First gameweek to analyze
            end_gw: Last gameweek to analyze (None = current GW)

        Returns:
            BacktestSummary with comparison results
        """
        logger.info(f"Running team backtest for team {self.team_id}")

        # Fetch historical data
        history = self.fetch_team_history()
        current_season = history['current']

        # Filter to requested gameweeks
        if end_gw is None:
            end_gw = len(current_season)

        gameweeks = []

        for gw_data in current_season:
            gw = gw_data['event']

            if gw < start_gw or gw > end_gw:
                continue

            # Get picks for this gameweek
            picks = self.fetch_gameweek_picks(gw)

            # Extract actual performance
            comparison = GameweekComparison(
                gameweek=gw,
                actual_points=gw_data['points'],
                actual_transfers=gw_data['event_transfers'],
                actual_transfer_cost=gw_data['event_transfers_cost'],
                actual_bench_points=gw_data['points_on_bench'],
            )

            # Extract captain info if available
            if picks and 'picks' in picks:
                for pick in picks['picks']:
                    if pick['is_captain']:
                        # Captain gets 2x points, so divide by 2 to get base points
                        # (This is approximate - we'd need player data to be exact)
                        comparison.actual_captain_points = pick.get('multiplier', 2)
                        break

            # Note: optimizer_points would be filled in by running optimizer
            # For now, we're just collecting actual data
            comparison.notes = f"Rank: {gw_data['rank']:,} | Overall: {gw_data['overall_rank']:,}"

            gameweeks.append(comparison)

        # Calculate summary
        summary = BacktestSummary(gameweeks=gameweeks)

        summary.actual_total_points = sum(gw.actual_points for gw in gameweeks)
        summary.actual_total_transfer_cost = sum(gw.actual_transfer_cost for gw in gameweeks)

        return summary

    def print_summary(self, summary: BacktestSummary):
        """Print a formatted summary of backtest results."""

        print("\n" + "=" * 80)
        print(f"FPL TEAM BACKTEST - Team ID: {self.team_id}")
        print("=" * 80)

        print(f"\n📊 GAMEWEEK-BY-GAMEWEEK PERFORMANCE")
        print("-" * 80)
        print(f"{'GW':<4} {'Points':<8} {'Transfers':<12} {'Hit':<6} {'Bench':<8} {'Notes':<30}")
        print("-" * 80)

        for gw in summary.gameweeks:
            transfers_str = f"{gw.actual_transfers} (-{gw.actual_transfer_cost})" if gw.actual_transfer_cost > 0 else str(gw.actual_transfers)
            hit_str = f"-{gw.actual_transfer_cost}" if gw.actual_transfer_cost > 0 else "-"

            print(f"{gw.gameweek:<4} {gw.actual_points:<8} {transfers_str:<12} {hit_str:<6} {gw.actual_bench_points:<8} {gw.notes:<30}")

        print("-" * 80)
        print(f"\n📈 SEASON SUMMARY")
        print("-" * 80)
        print(f"Total Points:        {summary.actual_total_points:,}")
        print(f"Gameweeks Played:    {len(summary.gameweeks)}")
        print(f"Average Points/GW:   {summary.actual_total_points / len(summary.gameweeks):.1f}")
        print(f"Total Transfer Cost: {summary.actual_total_transfer_cost}")
        print(f"Points Lost to Hits: {summary.actual_total_transfer_cost}")

        # Best and worst gameweeks
        best_gw = max(summary.gameweeks, key=lambda x: x.actual_points)
        worst_gw = min(summary.gameweeks, key=lambda x: x.actual_points)

        print(f"\nBest Gameweek:       GW{best_gw.gameweek} ({best_gw.actual_points} pts)")
        print(f"Worst Gameweek:      GW{worst_gw.gameweek} ({worst_gw.actual_points} pts)")

        # Bench analysis
        total_bench = sum(gw.actual_bench_points for gw in summary.gameweeks)
        print(f"\nTotal Bench Points:  {total_bench}")
        print(f"Avg Bench/GW:        {total_bench / len(summary.gameweeks):.1f}")

        print("\n" + "=" * 80)

        print("\n⚠️  NOTE: This backtest shows your actual performance.")
        print("To compare vs optimizer, we need to:")
        print("1. Simulate optimizer decisions week-by-week (avoiding hindsight bias)")
        print("2. Calculate what points those decisions would have earned")
        print("3. Compare side-by-side")
        print("\nThis requires running the full optimizer for each historical gameweek,")
        print("which is computationally expensive but possible!")
        print("=" * 80 + "\n")


def main():
    """Run team backtest."""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    team_id = int(os.getenv('FPL_TEAM_ID'))

    backtester = TeamBacktester(team_id)
    summary = backtester.run_backtest(start_gw=1)
    backtester.print_summary(summary)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()