#!/usr/bin/env python3
"""
Realistic Optimizer Backtest

Uses the actual SimpleExpectedPointsCalculator to make decisions:
- Squad selection based on expected points
- Captain selection based on expected points
- Transfer decisions based on expected points

Compares performance against:
- Your actual team
- Top 10k average
- Overall average

This is the real test of whether the optimizer beats humans.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict

from fpl_optimizer.fpl_client import FPLClient
from fpl_optimizer.models import Player, Team, Fixture
from fpl_optimizer.expected_points import SimpleExpectedPointsCalculator
from fpl_optimizer.data_cache import DataCache
from fpl_optimizer.data_sources import build_advanced_stats

logging.basicConfig(level=logging.INFO, format='%(message)s')
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
    BlendConfig(0.70, 0.30, 0.50, 0.50, 0.40, 0.60, "70/30 (Current)"),
]


class RealisticTeamManager:
    """
    Manages an FPL team using the actual expected points calculator.

    This is what would actually happen if you followed the optimizer's
    recommendations religiously.
    """

    def __init__(self, config: BlendConfig):
        self.config = config
        self.squad = []  # List of player IDs
        self.budget = 1000  # £100.0M in 0.1M units
        self.free_transfers = 1
        self.total_points = 0
        self.gw_results = []

        # Initialize calculator with custom blend ratios
        self.calculator = SimpleExpectedPointsCalculator(
            elite_predictive_weight=config.elite_predictive,
            elite_reactive_weight=config.elite_reactive,
            good_predictive_weight=config.good_predictive,
            good_reactive_weight=config.good_reactive,
            avg_predictive_weight=config.avg_predictive,
            avg_reactive_weight=config.avg_reactive,
        )

    def initialize(self, players: List[Player], teams: Dict[int, Team],
                   player_stats: Dict, team_strength: Dict):
        """Initialize the calculator with data."""
        self.players = {p.id: p for p in players}
        self.teams = teams
        self.calculator.load_data(player_stats, team_strength)

    def select_gw1_squad(self, fixtures: List[Fixture], gw: int) -> List[int]:
        """
        Select GW1 squad using the optimizer's expected points.

        This uses the ACTUAL optimizer logic, not a heuristic.
        """
        logger.info(f"\n🔨 Selecting GW1 Squad using {self.config.name}...")

        # Calculate expected points for all players
        player_xp = []
        for player in self.players.values():
            try:
                breakdown = self.calculator.calculate(player, gw, fixtures, self.teams)
                player_xp.append((player, breakdown.total_expected_points))
            except Exception as e:
                logger.warning(f"Failed to calculate xP for {player.name}: {e}")
                continue

        # Sort by value: xP per million
        player_xp.sort(key=lambda x: x[1] / x[0].cost_millions if x[0].cost_millions > 0 else 0,
                       reverse=True)

        # Greedy selection with FPL constraints
        squad = []
        budget = 1000  # £100M
        position_counts = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        position_limits = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        team_counts = defaultdict(int)

        for player, xp in player_xp:
            if len(squad) >= 15:
                break
            if player.cost > budget:
                continue
            if position_counts[player.position_name] >= position_limits[player.position_name]:
                continue
            if team_counts[player.team_id] >= 3:
                continue

            squad.append(player.id)
            budget -= player.cost
            position_counts[player.position_name] += 1
            team_counts[player.team_id] += 1

        self.budget = budget
        self.squad = squad

        logger.info(f"✓ Selected {len(squad)} players, £{budget/10:.1f}M remaining")
        logger.info(f"  Formation: {position_counts}")

        return squad

    def select_captain_and_lineup(self, fixtures: List[Fixture], gw: int) -> Tuple[List[int], int]:
        """
        Select captain and starting 11 using expected points.

        Captain = highest expected points
        Starting 11 = top 11 by expected points (respecting formation rules)
        """
        # Calculate expected points for squad
        squad_xp = []
        for pid in self.squad:
            player = self.players.get(pid)
            if not player:
                continue

            try:
                breakdown = self.calculator.calculate(player, gw, fixtures, self.teams)
                squad_xp.append((player, breakdown.total_expected_points))
            except Exception:
                squad_xp.append((player, 0.0))

        # Sort by expected points
        squad_xp.sort(key=lambda x: x[1], reverse=True)

        # Captain = highest xP
        captain_id = squad_xp[0][0].id if squad_xp else None

        # Select starting 11 (1 GKP, 3+ DEF, 2+ MID, 1+ FWD)
        starting = []
        position_counts = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}

        # First pass: ensure minimums
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            min_count = {'GKP': 1, 'DEF': 3, 'MID': 2, 'FWD': 1}[pos]
            pos_players = [(p, xp) for p, xp in squad_xp if p.position_name == pos]

            for player, xp in pos_players[:min_count]:
                if len(starting) < 11:
                    starting.append(player.id)
                    position_counts[pos] += 1

        # Fill remaining slots with highest xP
        for player, xp in squad_xp:
            if len(starting) >= 11:
                break
            if player.id in starting:
                continue

            pos = player.position_name
            # Check formation limits (max 5 DEF, 5 MID, 3 FWD)
            max_limits = {'GKP': 1, 'DEF': 5, 'MID': 5, 'FWD': 3}
            if position_counts[pos] < max_limits[pos]:
                starting.append(player.id)
                position_counts[pos] += 1

        return starting, captain_id


def main():
    """
    Run realistic backtest showing what would happen if you
    followed the optimizer's recommendations.
    """
    print("\n" + "="*80)
    print("REALISTIC OPTIMIZER BACKTEST")
    print("="*80)
    print("\nThis test simulates what would happen if you followed")
    print("the optimizer's recommendations religiously from GW1.")
    print("\nUsing:")
    print("  - Full SimpleExpectedPointsCalculator")
    print("  - Actual xG/xA data")
    print("  - Realistic transfer strategy")
    print("  - Captain picks based on expected points")
    print("\n" + "="*80)

    # Initialize FPL client
    print("\n[1/5] Fetching current FPL data...")
    client = FPLClient()
    bootstrap = client.get_bootstrap_static()

    players = [Player(**p) for p in bootstrap['elements']]
    teams_data = [Team(**t) for t in bootstrap['teams']]
    teams = {t.id: t for t in teams_data}

    print(f"  ✓ Loaded {len(players)} players, {len(teams)} teams")

    # Load cached advanced stats
    print("\n[2/5] Loading advanced stats...")
    cache = DataCache()
    player_stats = cache.load_player_stats()
    team_strength = cache.load_team_strength()

    if not player_stats or not team_strength:
        print("  Cache miss - fetching external data...")
        player_stats, team_strength = build_advanced_stats(
            players, teams, use_understat=True, use_fbref=False
        )
    else:
        print(f"  ✓ Loaded cached stats for {len(player_stats)} players")

    # Get fixtures for GW1
    print("\n[3/5] Fetching fixtures...")
    fixtures_data = client.get_fixtures()
    fixtures = [Fixture(**f) for f in fixtures_data if f.get('event') == 1]
    print(f"  ✓ Loaded {len(fixtures)} GW1 fixtures")

    # Initialize manager
    print("\n[4/5] Initializing team manager...")
    config = BLEND_CONFIGS[0]  # 70/30 (current optimal)
    manager = RealisticTeamManager(config)
    manager.initialize(players, teams, player_stats, team_strength)
    print(f"  ✓ Using {config.name} blend ratio")

    # Select GW1 squad
    print("\n[5/5] Selecting optimal GW1 squad...")
    squad = manager.select_gw1_squad(fixtures, gw=1)

    # Show selected squad
    print("\n" + "="*80)
    print("SELECTED GW1 SQUAD")
    print("="*80)
    print(f"\n{'Player':<25} {'Team':<15} {'Pos':<5} {'Cost':<8} {'xP (GW1)'}")
    print("-"*80)

    for pid in squad:
        player = manager.players[pid]
        try:
            breakdown = manager.calculator.calculate(player, 1, fixtures, teams)
            xp = breakdown.total_expected_points
        except:
            xp = 0.0

        team_name = teams[player.team_id].short_name
        print(f"{player.name:<25} {team_name:<15} {player.position_name:<5} £{player.cost_millions:<7.1f} {xp:.1f}")

    # Select captain and lineup
    starting, captain_id = manager.select_captain_and_lineup(fixtures, gw=1)
    captain_name = manager.players[captain_id].name if captain_id else "Unknown"

    print(f"\nCaptain: {captain_name}")
    print(f"Budget remaining: £{manager.budget/10:.1f}M")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\nTo complete this backtest, we need to:")
    print("  1. Fetch historical fixtures for each gameweek")
    print("  2. Simulate transfer decisions using expected points")
    print("  3. Calculate actual points scored each GW")
    print("  4. Compare to your actual performance (744 pts)")
    print("  5. Compare to top 10k average (~900-1000 pts)")
    print("\nThis would prove whether the optimizer beats human managers.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()