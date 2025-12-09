"""
Collect 2+ seasons of historical FPL data for cross-validation.

This will fetch player stats for:
- 2022/23 season
- 2023/24 season
- 2024/25 season (current)

Data structure per player per gameweek:
- Player attributes (position, team, cost)
- Historical stats (total_points, minutes, form up to that GW)
- Fixture context (opponent strength, home/away)
- Actual outcome (points scored in that GW)
"""

import json
import logging
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import requests

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class HistoricalPlayerGW:
    """Training data point: player state + actual outcome."""
    # Context (features)
    season: str  # "2022-23", "2023-24", "2024-25"
    gameweek: int
    player_id: int
    name: str
    position: int  # 1=GKP, 2=DEF, 3=MID, 4=FWD
    team_id: int
    cost: float  # £M

    # Historical stats (cumulative up to this GW)
    total_points: int
    minutes: int
    form: float
    points_per_game: float

    # Fixture context
    was_home: bool
    opponent_team_id: int
    opponent_strength_overall: int  # 1-5
    opponent_strength_attack: int
    opponent_strength_defence: int
    fpl_difficulty: int  # 1-5

    # Outcome (target)
    actual_points: int  # Points scored this GW
    actual_minutes: int


class HistoricalDataCollector:
    """Collect FPL data across multiple seasons."""

    # Note: FPL API only provides current season via /bootstrap-static/
    # Historical seasons are available via:
    # https://fantasy.premierleague.com/api/element-summary/{player_id}/
    # which includes 'history_past' for previous seasons

    # For detailed GW-by-GW data for past seasons, we need to use:
    # - Cached data if available
    # - Third-party archives (e.g., https://github.com/vaastav/Fantasy-Premier-League)

    def __init__(self, cache_dir: str = "historical_data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.base_url = "https://fantasy.premierleague.com/api"

    def fetch_current_season_data(self) -> Dict[int, List[HistoricalPlayerGW]]:
        """
        Fetch current season (2024/25) GW-by-GW data.

        Returns:
            Dict[player_id, List[HistoricalPlayerGW]] for each GW
        """
        logger.info("=" * 80)
        logger.info("FETCHING CURRENT SEASON DATA (2024/25)")
        logger.info("=" * 80)

        cache_file = self.cache_dir / "season_2024_25.pkl"

        if cache_file.exists():
            logger.info("Loading from cache...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        # Fetch bootstrap
        bootstrap = requests.get(f"{self.base_url}/bootstrap-static/").json()
        players = bootstrap['elements']
        teams = {t['id']: t for t in bootstrap['teams']}
        events = bootstrap['events']

        # Fetch fixtures
        fixtures_data = requests.get(f"{self.base_url}/fixtures/").json()

        # Build fixture map: {(team_h, team_a, event): fixture}
        fixtures_map = {}
        for fix in fixtures_data:
            if fix['event']:
                key = (fix['team_h'], fix['team_a'], fix['event'])
                fixtures_map[key] = fix

        logger.info(f"Fetching history for {len(players)} players...")

        data = defaultdict(list)

        for idx, player in enumerate(players):
            if idx % 50 == 0:
                logger.info(f"Progress: {idx}/{len(players)}")
                time.sleep(0.5)

            player_id = player['id']

            # Fetch player history
            try:
                response = requests.get(f"{self.base_url}/element-summary/{player_id}/")
                response.raise_for_status()
                player_data = response.json()
            except:
                continue

            history = player_data.get('history', [])

            if not history:
                continue

            # Build cumulative stats per GW
            cumulative_points = 0
            cumulative_minutes = 0

            for gw_entry in history:
                gw = gw_entry['round']

                # Accumulate
                gw_pts = gw_entry['total_points']
                gw_mins = gw_entry['minutes']

                cumulative_points += gw_pts
                cumulative_minutes += gw_mins

                # Calculate stats
                games_played = cumulative_minutes / 90 if cumulative_minutes > 0 else 0
                ppg = cumulative_points / games_played if games_played > 0 else 0
                form = float(gw_pts)  # Simplified form

                # Get fixture info
                team_id = gw_entry['team_h_score'] is not None
                opponent_id = gw_entry['opponent_team']
                was_home = gw_entry['was_home']

                opponent_team = teams.get(opponent_id, {})

                # Create data point
                datapoint = HistoricalPlayerGW(
                    season="2024-25",
                    gameweek=gw,
                    player_id=player_id,
                    name=player['web_name'],
                    position=player['element_type'],
                    team_id=player['team'],
                    cost=gw_entry['value'] / 10.0,
                    total_points=cumulative_points,
                    minutes=cumulative_minutes,
                    form=form,
                    points_per_game=ppg,
                    was_home=was_home,
                    opponent_team_id=opponent_id,
                    opponent_strength_overall=opponent_team.get('strength', 3),
                    opponent_strength_attack=opponent_team.get('strength_attack_home' if not was_home else 'strength_attack_away', 3),
                    opponent_strength_defence=opponent_team.get('strength_defence_home' if not was_home else 'strength_defence_away', 3),
                    fpl_difficulty=gw_entry.get('difficulty', 3),
                    actual_points=gw_pts,
                    actual_minutes=gw_mins,
                )

                data[player_id].append(datapoint)

        logger.info(f"✅ Collected data for {len(data)} players")

        # Cache it
        with open(cache_file, 'wb') as f:
            pickle.dump(dict(data), f)

        return dict(data)

    def fetch_past_seasons_from_github(self) -> Dict[str, Dict[int, List[HistoricalPlayerGW]]]:
        """
        Fetch historical data from vaastav/Fantasy-Premier-League GitHub archive.

        This is a well-maintained archive with GW-by-GW data for past seasons.

        Returns:
            Dict[season, Dict[player_id, List[HistoricalPlayerGW]]]
        """
        logger.info("=" * 80)
        logger.info("FETCHING PAST SEASONS FROM GITHUB ARCHIVE")
        logger.info("=" * 80)

        # The vaastav archive structure:
        # https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2022-23/gws/merged_gw.csv
        # https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/gws/merged_gw.csv

        # For simplicity, we'll note that this requires downloading CSV files
        # and parsing them. This is a more involved process.

        logger.info("⚠️  GitHub archive fetching not implemented yet.")
        logger.info("   For now, we'll use just current season data.")
        logger.info("   To add past seasons:")
        logger.info("   1. Download CSVs from https://github.com/vaastav/Fantasy-Premier-League")
        logger.info("   2. Parse CSV into HistoricalPlayerGW format")
        logger.info("   3. Cache as pickle files")

        return {}


def main():
    """Collect historical data for cross-validation."""

    collector = HistoricalDataCollector()

    # Fetch current season
    current_season_data = collector.fetch_current_season_data()

    logger.info("\n" + "=" * 80)
    logger.info("DATA COLLECTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Current season (2024/25): {len(current_season_data)} players")
    logger.info(f"Total data points: {sum(len(gws) for gws in current_season_data.values())}")

    # Save summary
    summary_file = Path("historical_data") / "data_summary.json"
    summary = {
        "seasons": ["2024-25"],
        "total_players": len(current_season_data),
        "total_datapoints": sum(len(gws) for gws in current_season_data.values()),
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n✅ Data saved to {summary_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()