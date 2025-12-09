"""
Full Historical Backtest

Simulates the optimizer managing a team from GW1 with real historical data.
Fetches gameweek-by-gameweek player performance and reconstructs what
decisions the optimizer would have made.
"""

import json
import logging
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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


class HistoricalDataBuilder:
    """Builds historical player database from FPL API."""

    def __init__(self, cache_dir: str = "backtest_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.base_url = "https://fantasy.premierleague.com/api"

    def fetch_bootstrap(self) -> Dict:
        """Fetch current bootstrap data."""
        cache_file = self.cache_dir / "bootstrap.json"

        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)

        logger.info("Fetching bootstrap data...")
        response = requests.get(f"{self.base_url}/bootstrap-static/")
        response.raise_for_status()
        data = response.json()

        with open(cache_file, 'w') as f:
            json.dump(data, f)

        return data

    def fetch_player_history(self, player_id: int) -> List[Dict]:
        """Fetch a player's gameweek history."""
        cache_file = self.cache_dir / f"player_{player_id}_history.json"

        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)

        try:
            response = requests.get(f"{self.base_url}/element-summary/{player_id}/")
            response.raise_for_status()
            data = response.json()
            history = data.get('history', [])

            with open(cache_file, 'w') as f:
                json.dump(history, f)

            return history
        except Exception as e:
            logger.warning(f"Could not fetch player {player_id}: {e}")
            return []

    def build_historical_database(self, max_gw: int = 14) -> Dict[int, Dict[int, PlayerGWData]]:
        """
        Build database of player stats at each gameweek.

        Returns:
            Dict[gameweek, Dict[player_id, PlayerGWData]]
        """
        logger.info("=" * 80)
        logger.info("BUILDING HISTORICAL DATABASE")
        logger.info("=" * 80)

        cache_file = self.cache_dir / f"historical_db_gw{max_gw}.pkl"

        if cache_file.exists():
            logger.info(f"Loading cached database...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        # Fetch bootstrap to get player list
        bootstrap = self.fetch_bootstrap()
        players = bootstrap['elements']
        teams_map = {t['id']: t['name'] for t in bootstrap['teams']}
        position_map = {p['id']: p['singular_name_short'] for p in bootstrap['element_types']}

        logger.info(f"Fetching history for {len(players)} players...")

        # Build database
        db: Dict[int, Dict[int, PlayerGWData]] = defaultdict(dict)

        for idx, player in enumerate(players):
            if idx % 50 == 0:
                logger.info(f"Progress: {idx}/{len(players)} players...")
                time.sleep(0.5)  # Rate limiting

            player_id = player['id']
            name = player['web_name']
            position = position_map[player['element_type']]
            team = teams_map[player['team']]

            # Fetch this player's history
            history = self.fetch_player_history(player_id)

            if not history:
                continue

            # Build cumulative stats for each GW
            cumulative_points = 0
            cumulative_minutes = 0

            for gw_entry in history:
                gw = gw_entry['round']

                if gw > max_gw:
                    break

                # Add this GW's points to cumulative
                gw_pts = gw_entry['total_points']
                gw_mins = gw_entry['minutes']

                cumulative_points += gw_pts
                cumulative_minutes += gw_mins

                # Calculate form (simple: last GW points)
                form = float(gw_pts)

                # Points per game
                games_played = cumulative_minutes / 90 if cumulative_minutes > 0 else 0
                ppg = cumulative_points / games_played if games_played > 0 else 0

                # Cost (in £M)
                cost = gw_entry['value'] / 10.0

                # Create PlayerGWData
                player_data = PlayerGWData(
                    player_id=player_id,
                    name=name,
                    position=position,
                    team=team,
                    cost=cost,
                    total_points=cumulative_points,
                    minutes=cumulative_minutes,
                    form=form,
                    points_per_game=ppg,
                    gw_points=gw_pts,
                    gw_minutes=gw_mins,
                )

                db[gw][player_id] = player_data

        logger.info(f"✅ Built database for {len(db)} gameweeks")

        # Cache it
        with open(cache_file, 'wb') as f:
            pickle.dump(dict(db), f)

        return dict(db)


class SimpleOptimizerBacktest:
    """
    Simplified optimizer backtest.

    Uses greedy heuristics instead of full LP optimization for speed.
    """

    def __init__(self, db: Dict[int, Dict[int, PlayerGWData]]):
        self.db = db

    def select_gw1_squad(self) -> List[int]:
        """
        Select optimal GW1 squad using greedy approach.

        Returns list of 15 player IDs.
        """
        logger.info("\n🔨 Selecting GW1 Squad...")

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
            # Check constraints
            if len(squad) >= 15:
                break
            if player.cost > budget:
                continue
            if position_counts[player.position] >= position_limits[player.position]:
                continue
            if team_counts[player.team] >= 3:
                continue

            # Add player
            squad.append(player.player_id)
            budget -= player.cost
            position_counts[player.position] += 1
            team_counts[player.team] += 1

        logger.info(f"✅ Selected {len(squad)} players, £{budget:.1f}M remaining")

        return squad

    def select_starting_11(self, squad_ids: List[int], gw: int) -> Tuple[List[int], int]:
        """
        Select starting 11 and captain from squad.

        Returns (starting_11_ids, captain_id)
        """
        players = [self.db[gw][pid] for pid in squad_ids if pid in self.db[gw]]

        # Sort by form/expected points
        players.sort(key=lambda p: p.form, reverse=True)

        # Formation: 1 GKP, 3-5 DEF, 3-5 MID, 1-3 FWD
        starting = []
        position_counts = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}

        # First pass: pick best player in each position
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            pos_players = [p for p in players if p.position == pos]
            if pos_players and pos == 'GKP':
                starting.append(pos_players[0].player_id)
                position_counts['GKP'] = 1

        # Pick 4 defenders
        def_players = [p for p in players if p.position == 'DEF']
        for p in def_players[:4]:
            if len(starting) < 11:
                starting.append(p.player_id)
                position_counts['DEF'] += 1

        # Pick 4 midfielders
        mid_players = [p for p in players if p.position == 'MID']
        for p in mid_players[:4]:
            if len(starting) < 11:
                starting.append(p.player_id)
                position_counts['MID'] += 1

        # Pick 2 forwards
        fwd_players = [p for p in players if p.position == 'FWD']
        for p in fwd_players[:2]:
            if len(starting) < 11:
                starting.append(p.player_id)
                position_counts['FWD'] += 1

        # Captain: best player by form
        captain_id = starting[0] if starting else squad_ids[0]

        return starting, captain_id

    def recommend_transfers(
        self,
        current_squad: List[int],
        gw: int,
        free_transfers: int
    ) -> List[Tuple[int, int]]:
        """
        Recommend transfers for this gameweek.

        Returns list of (player_out_id, player_in_id) tuples.
        """
        if free_transfers == 0:
            return []

        # Simple strategy: transfer out lowest performers
        current_players = [self.db[gw].get(pid) for pid in current_squad]
        current_players = [p for p in current_players if p is not None]

        # Sort by form (ascending)
        current_players.sort(key=lambda p: p.form)

        transfers = []

        # Transfer out worst 1-2 players
        num_transfers = min(free_transfers, 2)

        for i in range(num_transfers):
            if i >= len(current_players):
                break

            player_out = current_players[i]

            # Find replacement
            all_players = list(self.db[gw].values())
            all_players.sort(key=lambda p: p.form, reverse=True)

            for player_in in all_players:
                if player_in.player_id in current_squad:
                    continue
                if player_in.position != player_out.position:
                    continue
                if player_in.cost > player_out.cost + 0.5:  # Max 0.5M upgrade
                    continue

                transfers.append((player_out.player_id, player_in.player_id))
                current_squad.remove(player_out.player_id)
                current_squad.append(player_in.player_id)
                break

        return transfers

    def calculate_gw_points(
        self,
        starting_11: List[int],
        captain_id: int,
        gw: int
    ) -> Tuple[int, int]:
        """
        Calculate points scored this gameweek.

        Returns (total_points, bench_points)
        """
        total = 0

        for pid in starting_11:
            if pid not in self.db[gw]:
                continue

            player = self.db[gw][pid]
            pts = player.gw_points

            # Captain gets 2x
            if pid == captain_id:
                pts *= 2

            total += pts

        return total, 0  # Simplified: no bench points tracked

    def run_backtest(self, start_gw: int = 1, end_gw: int = 14):
        """Run full backtest simulation."""
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING OPTIMIZER BACKTEST")
        logger.info("=" * 80)

        # GW1: Build squad
        squad = self.select_gw1_squad()
        free_transfers = 1
        total_points = 0
        total_hits = 0

        results = []

        for gw in range(start_gw, end_gw + 1):
            logger.info(f"\n📅 GW{gw}")
            logger.info("-" * 40)

            # Make transfers (if not GW1)
            transfers = []
            hit_cost = 0

            if gw > 1:
                transfers = self.recommend_transfers(squad, gw, free_transfers)

                if len(transfers) > free_transfers:
                    hit_cost = (len(transfers) - free_transfers) * 4
                    total_hits += hit_cost

                logger.info(f"Transfers: {len(transfers)} (cost: {hit_cost})")

                # Apply transfers
                for out_id, in_id in transfers:
                    if out_id in squad:
                        squad.remove(out_id)
                    squad.append(in_id)

            # Select starting 11 and captain
            starting_11, captain_id = self.select_starting_11(squad, gw)

            # Calculate points
            gw_points, bench_pts = self.calculate_gw_points(starting_11, captain_id, gw)
            gw_points -= hit_cost

            total_points += gw_points

            logger.info(f"Points: {gw_points} (hits: -{hit_cost})")

            # Update free transfers
            if len(transfers) >= free_transfers:
                free_transfers = 1
            else:
                free_transfers = min(2, free_transfers + 1)

            results.append((gw, gw_points, hit_cost))

        return results, total_points, total_hits


def main():
    """Run full backtest."""

    # Step 1: Build historical database
    builder = HistoricalDataBuilder()
    db = builder.build_historical_database(max_gw=14)

    # Step 2: Run backtest
    backtester = SimpleOptimizerBacktest(db)
    results, total_points, total_hits = backtester.run_backtest(1, 14)

    # Step 3: Compare vs actual
    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)

    logger.info(f"\n{'GW':<4} {'Optimizer Pts':<15} {'Hit Cost':<10}")
    logger.info("-" * 40)

    for gw, pts, hit in results:
        logger.info(f"{gw:<4} {pts:<15} {hit:<10}")

    logger.info("-" * 40)
    logger.info(f"\nOptimizer Total: {total_points} points")
    logger.info(f"Total Hits:      {total_hits} points")
    logger.info(f"Net Score:       {total_points} points")

    logger.info(f"\n📊 COMPARISON")
    logger.info("-" * 40)
    logger.info(f"Your Actual:     776 points (-32 hits = 744 net)")
    logger.info(f"Optimizer:       {total_points} points (-{total_hits} hits = {total_points} net)")
    logger.info(f"Difference:      +{total_points - 744} points")

    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()