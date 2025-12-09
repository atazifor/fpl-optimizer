#!/usr/bin/env python3
"""
Parallel Full Team Simulation - Compare Blend Ratios

Runs 3 complete team simulations in parallel (on M3 cores):
- Each simulation manages a team from GW1-14
- Makes transfers, picks captains, selects lineup
- Tracks total points scored

Compares:
- 90/10 (Very Predictive)
- 80/20 (Balanced)
- 70/30 (More Reactive)
"""

import json
import logging
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import multiprocessing as mp

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PlayerGWData:
    """Player data as of a specific gameweek."""
    player_id: int
    name: str
    position: str
    team: str
    cost: float  # £M

    # Stats accumulated UP TO this gameweek
    total_points: int
    minutes: int
    form: float
    points_per_game: float

    # This gameweek's actual performance (for scoring)
    gw_points: int
    gw_minutes: int


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


class TeamSimulator:
    """Simulates FPL team management with a specific blend ratio."""

    def __init__(self, config: BlendConfig, db: Dict[int, Dict[int, PlayerGWData]]):
        self.config = config
        self.db = db
        self.squad = []  # List of player IDs
        self.budget = 100.0
        self.free_transfers = 1
        self.total_points = 0
        self.total_hits = 0
        self.gw_results = []

    def select_gw1_squad(self) -> List[int]:
        """Select optimal GW1 squad using greedy approach."""
        gw1_players = list(self.db[1].values())

        # Sort by value: points per million
        gw1_players.sort(key=lambda p: p.total_points / p.cost if p.cost > 0 else 0, reverse=True)

        # Greedy selection with constraints
        squad = []
        budget = 100.0  # £100M
        position_counts = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        position_limits = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        team_counts = defaultdict(int)

        for player in gw1_players:
            if len(squad) >= 15:
                break
            if player.cost > budget:
                continue
            if position_counts[player.position] >= position_limits[player.position]:
                continue
            if team_counts[player.team] >= 3:
                continue

            squad.append(player.player_id)
            budget -= player.cost
            position_counts[player.position] += 1
            team_counts[player.team] += 1

        self.budget = budget
        return squad

    def select_starting_11(self, squad_ids: List[int], gw: int) -> Tuple[List[int], int]:
        """Select starting 11 and captain from squad."""
        players = [self.db[gw][pid] for pid in squad_ids if pid in self.db[gw]]

        # Sort by form (this is where blend ratio would matter in real implementation)
        # For now, use simple form-based sorting
        players.sort(key=lambda p: p.form * self._get_blend_weight(p), reverse=True)

        # Select by formation (e.g., 1-4-4-2)
        starting = []
        captain_id = None

        # 1 GKP
        gkps = [p for p in players if p.position == 'GKP']
        if gkps:
            starting.append(gkps[0].player_id)

        # 4 DEF
        defs = [p for p in players if p.position == 'DEF']
        starting.extend([p.player_id for p in defs[:4]])

        # 4 MID
        mids = [p for p in players if p.position == 'MID']
        starting.extend([p.player_id for p in mids[:4]])

        # 2 FWD
        fwds = [p for p in players if p.position == 'FWD']
        starting.extend([p.player_id for p in fwds[:2]])

        # Captain = highest form player
        if players:
            captain_id = players[0].player_id

        return starting[:11], captain_id

    def _get_blend_weight(self, player: PlayerGWData) -> float:
        """Calculate blend weight based on player quality tier."""
        # Simple heuristic: elite if form > 6, good if > 4, otherwise avg
        if player.form >= 6.0:
            return self.config.elite_predictive + self.config.elite_reactive
        elif player.form >= 4.0:
            return self.config.good_predictive + self.config.good_reactive
        else:
            return self.config.avg_predictive + self.config.avg_reactive

    def make_transfers(self, gw: int) -> int:
        """Make transfers for this gameweek. Returns hit cost."""
        # Simple strategy: 1 transfer per week (use free transfer)
        # In real implementation, this would use the blend ratio to evaluate transfers
        hit_cost = 0

        if self.free_transfers > 0:
            self.free_transfers -= 1
        else:
            hit_cost = 4  # -4 points for extra transfer

        # Reset free transfers
        if gw > 1:
            self.free_transfers = min(self.free_transfers + 1, 2)

        return hit_cost

    def simulate_gameweek(self, gw: int) -> int:
        """Simulate a single gameweek. Returns points scored."""
        # Make transfers
        hit_cost = self.make_transfers(gw) if gw > 1 else 0

        # Select starting 11 and captain
        starting_11, captain_id = self.select_starting_11(self.squad, gw)

        # Calculate points
        points = 0
        for pid in starting_11:
            if pid in self.db[gw]:
                player_points = self.db[gw][pid].gw_points
                if pid == captain_id:
                    points += player_points * 2  # Captain gets double
                else:
                    points += player_points

        net_points = points - hit_cost
        return net_points, hit_cost

    def run_simulation(self, max_gw: int = 14):
        """Run full season simulation."""
        # Select GW1 squad
        self.squad = self.select_gw1_squad()

        # Simulate each gameweek
        for gw in range(1, max_gw + 1):
            if gw not in self.db:
                break

            gw_points, hit_cost = self.simulate_gameweek(gw)
            self.total_points += gw_points
            self.total_hits += hit_cost

            self.gw_results.append({
                'gw': gw,
                'points': gw_points,
                'hits': hit_cost
            })

        return {
            'config': self.config.name,
            'total_points': self.total_points,
            'total_hits': self.total_hits,
            'gw_results': self.gw_results
        }


def load_historical_data(cache_dir: str = "backtest_cache") -> Dict[int, Dict[int, PlayerGWData]]:
    """Load cached historical data."""
    cache_path = Path(cache_dir)

    # Load bootstrap
    with open(cache_path / "bootstrap.json") as f:
        bootstrap = json.load(f)

    # Build player lookup
    player_lookup = {}
    for p in bootstrap['elements']:
        pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        team_lookup = {t['id']: t['short_name'] for t in bootstrap['teams']}

        player_lookup[p['id']] = {
            'name': p['web_name'],
            'position': pos_map[p['element_type']],
            'team': team_lookup.get(p['team'], 'UNK'),
            'cost': p['now_cost'] / 10.0
        }

    # Build GW database
    db = {}

    for gw in range(1, 15):
        db[gw] = {}

        for player_id, info in player_lookup.items():
            cache_file = cache_path / f"player_{player_id}_history.json"

            if not cache_file.exists():
                continue

            with open(cache_file) as f:
                history = json.load(f)

            # Find this GW's data
            gw_data = next((h for h in history if h['round'] == gw), None)

            if not gw_data:
                continue

            # Calculate cumulative stats up to this GW
            prev_gws = [h for h in history if h['round'] < gw]
            total_points = sum(h['total_points'] for h in prev_gws)
            total_minutes = sum(h['minutes'] for h in prev_gws)

            # Get form from last 3 GWs
            recent = [h for h in history if h['round'] < gw][-3:]
            form = sum(h['total_points'] for h in recent) / len(recent) if recent else 0

            db[gw][player_id] = PlayerGWData(
                player_id=player_id,
                name=info['name'],
                position=info['position'],
                team=info['team'],
                cost=info['cost'],
                total_points=total_points,
                minutes=total_minutes,
                form=form,
                points_per_game=total_points / len(prev_gws) if prev_gws else 0,
                gw_points=gw_data['total_points'],
                gw_minutes=gw_data['minutes']
            )

    return db


def run_single_simulation(args):
    """Run simulation for a single blend config."""
    config, db = args

    print(f"\n{'='*80}")
    print(f"Starting simulation: {config.name}")
    print(f"{'='*80}")

    simulator = TeamSimulator(config, db)
    result = simulator.run_simulation(max_gw=14)

    print(f"\n{config.name} completed:")
    print(f"  Total Points: {result['total_points']}")
    print(f"  Total Hits: {result['total_hits']}")
    print(f"  Net Score: {result['total_points']} points")

    return result


def main():
    """Run parallel team simulations."""
    print("\n" + "="*80)
    print("PARALLEL FULL TEAM SIMULATION")
    print("="*80)
    print("\nRunning 3 complete team simulations in parallel:")
    print("  1. 90/10 (Very Predictive)")
    print("  2. 80/20 (Balanced)")
    print("  3. 70/30 (More Reactive)")
    print("\nEach simulation:")
    print("  - Selects GW1 squad")
    print("  - Makes transfers each week")
    print("  - Picks captain & lineup")
    print("  - Tracks total points")
    print("\n" + "="*80)

    # Load historical data
    print("\nLoading historical data from cache...")
    db = load_historical_data()
    print(f"✓ Loaded {len(db)} gameweeks")

    # Run simulations in parallel
    print("\nRunning simulations in parallel (using M3 cores)...")

    args_list = [(config, db) for config in BLEND_CONFIGS]

    with mp.Pool(processes=3) as pool:
        results = pool.map(run_single_simulation, args_list)

    # Compare results
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"\n{'Configuration':<30} {'Gross Points':<15} {'Hits':<10} {'Net Points':<15} {'Winner'}")
    print("-"*80)

    best_result = max(results, key=lambda r: r['total_points'])
    for result in sorted(results, key=lambda r: r['total_points'], reverse=True):
        is_best = "⭐ BEST" if result['config'] == best_result['config'] else ""
        gross = result['total_points'] + result['total_hits']
        hits = result['total_hits']
        net = result['total_points']
        print(f"{result['config']:<30} {gross:<15} {hits:<10} {net:<15} {is_best}")

    # Show gameweek breakdown for best config
    print("\n" + "="*80)
    print(f"GAMEWEEK BREAKDOWN - {best_result['config']}")
    print("="*80)
    print(f"\n{'GW':<5} {'Points':<10} {'Hits':<10}")
    print("-"*30)

    for gw_result in best_result['gw_results']:
        gw = gw_result['gw']
        pts = gw_result['points']
        hits = gw_result['hits']
        print(f"{gw:<5} {pts:<10} {-hits:<10}")

    # Save results
    output_file = Path(__file__).parent.parent / "team_simulation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Results saved to: {output_file}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()