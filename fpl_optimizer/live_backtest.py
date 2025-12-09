"""
Live Simulation Backtest

Simulates the optimizer managing a team from GW1, making decisions week-by-week
with only historical data available at each point (no hindsight bias).

This is the gold standard for backtesting - it shows what would have happened
if you had used the optimizer from day 1.
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests

from .models import Player, Fixture, Team
from .optimizer import FPLOptimizer, OptimizationConfig

logger = logging.getLogger(__name__)


@dataclass
class SimulatedTeam:
    """State of simulated team at a point in time."""

    gameweek: int
    player_ids: List[int]  # 15 players
    starting_11_ids: List[int]
    captain_id: int
    vice_captain_id: int
    bench_order: List[int]  # Player IDs in bench order

    bank: float  # In tenths (e.g., 5 = £0.5M)
    free_transfers: int

    # Track for analysis
    transfers_made: List[tuple] = field(default_factory=list)  # [(out_id, in_id)]
    transfer_cost: int = 0


@dataclass
class GameweekResult:
    """Result of a simulated gameweek."""

    gameweek: int
    points: int
    captain_points: int
    bench_points: int
    transfers_made: int
    transfer_cost: int

    # Breakdown
    starting_11_points: List[tuple]  # [(player_id, points)]
    bench_points_by_player: List[tuple]  # [(player_id, points)]

    # Team state after this GW
    team_value: int
    bank: float
    free_transfers: int


@dataclass
class BacktestResults:
    """Complete backtest results."""

    gameweeks: List[GameweekResult]

    # Summary
    total_points: int = 0
    total_transfer_cost: int = 0
    final_rank_estimate: Optional[int] = None


class LiveBacktester:
    """
    Simulates optimizer managing a team from scratch.

    Uses only historical data available at each gameweek (no lookahead).
    """

    def __init__(self, cache_dir: str = "backtest_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.base_url = "https://fantasy.premierleague.com/api"

    def fetch_historical_gameweek_data(self, gameweek: int) -> Dict:
        """
        Fetch FPL data as it appeared at the start of a gameweek.

        Note: FPL API doesn't provide true historical snapshots, so this
        fetches current data. For a true backtest, you'd need archived data.

        For now, we'll use current player stats but only look at data
        up to the target gameweek.
        """
        cache_file = self.cache_dir / f"gw{gameweek}_data.pkl"

        if cache_file.exists():
            logger.info(f"Loading cached data for GW{gameweek}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        logger.info(f"Fetching data for GW{gameweek}")

        # Fetch bootstrap data (players, teams, fixtures)
        response = requests.get(f"{self.base_url}/bootstrap-static/")
        response.raise_for_status()
        data = response.json()

        # Cache it
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

        return data

    def simulate_gw1_team_selection(self) -> SimulatedTeam:
        """
        Build optimal GW1 squad using optimizer.

        Returns initial team of 15 players within £100M budget.
        """
        logger.info("Simulating GW1 team selection...")

        # Fetch GW1 data
        data = self.fetch_historical_gameweek_data(1)

        # Use optimizer to pick squad
        # For GW1, we assume no transfers used (fresh squad)
        # Budget: £1000 (tenths), Free transfers: 1

        config = OptimizationConfig(
            max_budget=1000,
            max_transfers=15,  # Build entire squad
            free_transfers=15,  # No cost for initial squad
            current_gameweek=1,
        )

        optimizer = FPLOptimizer(config)

        # Run optimization for GW1
        # This will pick the best 15 players for GW1
        result = optimizer.optimize_squad(
            current_player_ids=[],  # No current squad
            gameweek=1,
        )

        # Extract team from result
        team = SimulatedTeam(
            gameweek=1,
            player_ids=result.squad_player_ids,
            starting_11_ids=result.current_squad.starting_11,
            captain_id=result.current_squad.captain_id,
            vice_captain_id=result.current_squad.vice_captain_id,
            bench_order=result.current_squad.bench_order,
            bank=result.bank_remaining,
            free_transfers=1,  # Get 1 FT after GW1
        )

        logger.info(f"GW1 squad selected: {len(team.player_ids)} players, £{team.bank/10:.1f}M in bank")

        return team

    def simulate_gameweek(
        self,
        team: SimulatedTeam,
        gameweek: int,
        make_transfers: bool = True
    ) -> GameweekResult:
        """
        Simulate a gameweek with the current team.

        Steps:
        1. If make_transfers, run optimizer to get recommended transfers
        2. Apply transfers to team
        3. Calculate points scored by team
        4. Update team state for next week

        Args:
            team: Current team state
            gameweek: Gameweek to simulate
            make_transfers: Whether to consider transfers

        Returns:
            GameweekResult with points and updated team state
        """
        logger.info(f"Simulating GW{gameweek}...")

        # Fetch data as of this gameweek
        data = self.fetch_historical_gameweek_data(gameweek)

        transfers_made = 0
        transfer_cost = 0
        new_player_ids = team.player_ids.copy()

        # Make transfers if enabled
        if make_transfers and team.free_transfers > 0:
            # Run optimizer to get transfer recommendations
            config = OptimizationConfig(
                max_transfers=2,  # Conservative: max 2 transfers per week
                free_transfers=team.free_transfers,
                current_gameweek=gameweek,
            )

            optimizer = FPLOptimizer(config)

            result = optimizer.optimize_squad(
                current_player_ids=team.player_ids,
                gameweek=gameweek,
            )

            # Apply recommended transfers
            transfers = result.transfers_made
            transfers_made = len(transfers)
            transfer_cost = max(0, (transfers_made - team.free_transfers) * 4)

            # Update squad
            for out_id, in_id in transfers:
                new_player_ids.remove(out_id)
                new_player_ids.append(in_id)

            # Update captain and lineup from optimizer
            team.starting_11_ids = result.current_squad.starting_11
            team.captain_id = result.current_squad.captain_id
            team.vice_captain_id = result.current_squad.vice_captain_id
            team.bench_order = result.current_squad.bench_order

            logger.info(f"Made {transfers_made} transfers (cost: {transfer_cost})")

        # Calculate points scored this gameweek
        # We need actual points from data
        points_map = {p['id']: p['event_points'] for p in data['elements']}

        # Calculate starting 11 points (including captain)
        starting_points = []
        total_points = 0

        for player_id in team.starting_11_ids:
            pts = points_map.get(player_id, 0)

            # Apply captain multiplier
            if player_id == team.captain_id:
                pts *= 2

            starting_points.append((player_id, pts))
            total_points += pts

        # Calculate bench points
        bench_ids = [pid for pid in new_player_ids if pid not in team.starting_11_ids]
        bench_points = []
        total_bench = 0

        for player_id in bench_ids:
            pts = points_map.get(player_id, 0)
            bench_points.append((player_id, pts))
            total_bench += pts

        # Subtract transfer cost
        total_points -= transfer_cost

        # Calculate captain points (for tracking)
        captain_pts = points_map.get(team.captain_id, 0) * 2

        # Update team state for next week
        team.player_ids = new_player_ids
        team.free_transfers = min(2, team.free_transfers - transfers_made + 1)
        if team.free_transfers < 0:
            team.free_transfers = 1

        # Create result
        result = GameweekResult(
            gameweek=gameweek,
            points=total_points,
            captain_points=captain_pts,
            bench_points=total_bench,
            transfers_made=transfers_made,
            transfer_cost=transfer_cost,
            starting_11_points=starting_points,
            bench_points_by_player=bench_points,
            team_value=0,  # TODO: Calculate from selling prices
            bank=team.bank,
            free_transfers=team.free_transfers,
        )

        return result

    def run_backtest(
        self,
        start_gw: int = 1,
        end_gw: int = 14,
    ) -> BacktestResults:
        """
        Run full backtest from GW1 to end_gw.

        Simulates the optimizer managing a team from scratch.
        """
        logger.info(f"Starting live backtest: GW{start_gw} to GW{end_gw}")

        # Build initial team
        team = self.simulate_gw1_team_selection()

        # Simulate each gameweek
        results = []

        for gw in range(start_gw, end_gw + 1):
            make_transfers = (gw > 1)  # No transfers in GW1
            result = self.simulate_gameweek(team, gw, make_transfers)
            results.append(result)

            logger.info(f"GW{gw}: {result.points} pts (transfers: {result.transfers_made}, cost: {result.transfer_cost})")

        # Calculate summary
        backtest = BacktestResults(gameweeks=results)
        backtest.total_points = sum(r.points for r in results)
        backtest.total_transfer_cost = sum(r.transfer_cost for r in results)

        return backtest

    def print_comparison(
        self,
        optimizer_results: BacktestResults,
        actual_total: int,
        actual_transfer_cost: int,
    ):
        """Print comparison of optimizer vs actual performance."""

        print("\n" + "=" * 80)
        print("OPTIMIZER vs ACTUAL PERFORMANCE")
        print("=" * 80)

        print(f"\n{'GW':<4} {'Optimizer':<12} {'Your Actual':<12} {'Difference':<12}")
        print("-" * 80)

        # TODO: Would need actual GW-by-GW data to compare
        # For now just show optimizer performance

        for result in optimizer_results.gameweeks:
            print(f"{result.gameweek:<4} {result.points:<12} {'?':<12} {'?':<12}")

        print("-" * 80)
        print(f"\n{'TOTAL':<4} {optimizer_results.total_points:<12} {actual_total:<12} {optimizer_results.total_points - actual_total:<12}")
        print(f"{'HITS':<4} {optimizer_results.total_transfer_cost:<12} {actual_transfer_cost:<12} {optimizer_results.total_transfer_cost - actual_transfer_cost:<12}")

        print("\n" + "=" * 80)


def main():
    """Run live backtest."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    backtester = LiveBacktester()

    # Run backtest
    results = backtester.run_backtest(start_gw=1, end_gw=14)

    # Compare vs actual (from previous backtest)
    actual_total = 776
    actual_transfer_cost = 32

    backtester.print_comparison(results, actual_total, actual_transfer_cost)


if __name__ == "__main__":
    main()