"""
Data Cache Module - Cache external stats to avoid re-scraping.

Caches:
- Understat player stats
- Understat team stats
- FBref data (if enabled)

Cache is stored in JSON files and expires after 24 hours.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

from .expected_points import PlayerStats, TeamStrength

logger = logging.getLogger(__name__)


class DataCache:
    """
    Cache for external data sources.

    Stores data in JSON files with timestamps to manage expiration.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize data cache.

        Args:
            cache_dir: Directory for cache files (default: ./.cache/fpl_optimizer)
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "fpl_optimizer"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.player_stats_file = self.cache_dir / "player_stats.json"
        self.team_strength_file = self.cache_dir / "team_strength.json"

        # Cache expires after 24 hours
        self.expiration_hours = 24

        logger.info(f"Data cache initialized at {self.cache_dir}")

    def is_cache_valid(self, filepath: Path) -> bool:
        """Check if cache file exists and hasn't expired."""
        if not filepath.exists():
            return False

        # Check modification time
        mod_time = datetime.fromtimestamp(filepath.stat().st_mtime)
        expiration_time = datetime.now() - timedelta(hours=self.expiration_hours)

        is_valid = mod_time > expiration_time

        if is_valid:
            age_hours = (datetime.now() - mod_time).total_seconds() / 3600
            logger.info(f"Cache valid: {filepath.name} (age: {age_hours:.1f}h)")
        else:
            logger.info(f"Cache expired: {filepath.name}")

        return is_valid

    def save_player_stats(self, player_stats: Dict[int, PlayerStats]) -> None:
        """
        Save player stats to cache.

        Args:
            player_stats: Dict mapping player_id -> PlayerStats
        """
        try:
            # Convert PlayerStats to dict for JSON serialization
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "data": {
                    str(player_id): {
                        "player_id": stats.player_id,
                        "position": stats.position,
                        "team_id": stats.team_id,
                        "xg_per_90": stats.xg_per_90,
                        "xa_per_90": stats.xa_per_90,
                        "xgi_per_90": stats.xgi_per_90,
                        "shots_per_90": stats.shots_per_90,
                        "shots_on_target_per_90": stats.shots_on_target_per_90,
                        "key_passes_per_90": stats.key_passes_per_90,
                        "big_chances_created_per_90": stats.big_chances_created_per_90,
                        "clean_sheet_percentage": stats.clean_sheet_percentage,
                        "saves_per_90": stats.saves_per_90,
                        "tackles_per_90": stats.tackles_per_90,
                        "interceptions_per_90": stats.interceptions_per_90,
                        "clearances_per_90": stats.clearances_per_90,
                        "is_penalty_taker": stats.is_penalty_taker,
                        "penalty_order": stats.penalty_order,
                        "penalties_taken": stats.penalties_taken,
                        "penalties_scored": stats.penalties_scored,
                        "is_set_piece_taker": stats.is_set_piece_taker,
                        "corners_per_90": stats.corners_per_90,
                        "direct_free_kicks_per_90": stats.direct_free_kicks_per_90,
                        "avg_minutes_per_game": stats.avg_minutes_per_game,
                        "games_started": stats.games_started,
                        "games_subbed_on": stats.games_subbed_on,
                        "games_subbed_off": stats.games_subbed_off,
                        "total_minutes": stats.total_minutes,
                        "avg_bonus_per_game": stats.avg_bonus_per_game,
                        "avg_bps_per_90": stats.avg_bps_per_90,
                        "form_xgi_per_90": stats.form_xgi_per_90,
                        "form_minutes": stats.form_minutes,
                        "form_games": stats.form_games,
                        "points_variance": stats.points_variance,
                        "blank_percentage": stats.blank_percentage,
                    }
                    for player_id, stats in player_stats.items()
                }
            }

            with open(self.player_stats_file, 'w') as f:
                json.dump(cache_data, f, indent=2)

            logger.info(f"Cached player stats for {len(player_stats)} players")

        except Exception as e:
            logger.error(f"Failed to save player stats cache: {e}")

    def load_player_stats(self) -> Optional[Dict[int, PlayerStats]]:
        """
        Load player stats from cache.

        Returns:
            Dict mapping player_id -> PlayerStats, or None if cache invalid
        """
        if not self.is_cache_valid(self.player_stats_file):
            return None

        try:
            with open(self.player_stats_file, 'r') as f:
                cache_data = json.load(f)

            # Convert dict back to PlayerStats objects
            player_stats = {}
            for player_id_str, stats_dict in cache_data["data"].items():
                player_id = int(player_id_str)
                player_stats[player_id] = PlayerStats(**stats_dict)

            logger.info(f"Loaded {len(player_stats)} player stats from cache")
            return player_stats

        except Exception as e:
            logger.error(f"Failed to load player stats cache: {e}")
            return None

    def save_team_strength(self, team_strength: Dict[int, TeamStrength]) -> None:
        """
        Save team strength to cache.

        Args:
            team_strength: Dict mapping team_id -> TeamStrength
        """
        try:
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "data": {
                    str(team_id): {
                        "team_id": strength.team_id,
                        "team_name": strength.team_name,
                        "xg_per_90": strength.xg_per_90,
                        "goals_per_90": strength.goals_per_90,
                        "shots_per_90": strength.shots_per_90,
                        "xga_per_90": strength.xga_per_90,
                        "goals_conceded_per_90": strength.goals_conceded_per_90,
                        "shots_conceded_per_90": strength.shots_conceded_per_90,
                        "home_xg_per_90": strength.home_xg_per_90,
                        "home_xga_per_90": strength.home_xga_per_90,
                        "away_xg_per_90": strength.away_xg_per_90,
                        "away_xga_per_90": strength.away_xga_per_90,
                        "clean_sheet_percentage": strength.clean_sheet_percentage,
                        "home_cs_percentage": strength.home_cs_percentage,
                        "away_cs_percentage": strength.away_cs_percentage,
                        "penalties_won_per_90": strength.penalties_won_per_90,
                        "penalties_conceded_per_90": strength.penalties_conceded_per_90,
                    }
                    for team_id, strength in team_strength.items()
                }
            }

            with open(self.team_strength_file, 'w') as f:
                json.dump(cache_data, f, indent=2)

            logger.info(f"Cached team strength for {len(team_strength)} teams")

        except Exception as e:
            logger.error(f"Failed to save team strength cache: {e}")

    def load_team_strength(self) -> Optional[Dict[int, TeamStrength]]:
        """
        Load team strength from cache.

        Returns:
            Dict mapping team_id -> TeamStrength, or None if cache invalid
        """
        if not self.is_cache_valid(self.team_strength_file):
            return None

        try:
            with open(self.team_strength_file, 'r') as f:
                cache_data = json.load(f)

            # Convert dict back to TeamStrength objects
            team_strength = {}
            for team_id_str, strength_dict in cache_data["data"].items():
                team_id = int(team_id_str)
                team_strength[team_id] = TeamStrength(**strength_dict)

            logger.info(f"Loaded {len(team_strength)} team strengths from cache")
            return team_strength

        except Exception as e:
            logger.error(f"Failed to load team strength cache: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear all cached data."""
        try:
            if self.player_stats_file.exists():
                self.player_stats_file.unlink()
                logger.info("Cleared player stats cache")

            if self.team_strength_file.exists():
                self.team_strength_file.unlink()
                logger.info("Cleared team strength cache")

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    def get_cache_info(self) -> Dict[str, any]:
        """Get information about the cache status."""
        info = {
            "cache_dir": str(self.cache_dir),
            "player_stats": {
                "exists": self.player_stats_file.exists(),
                "valid": self.is_cache_valid(self.player_stats_file),
                "age_hours": None,
            },
            "team_strength": {
                "exists": self.team_strength_file.exists(),
                "valid": self.is_cache_valid(self.team_strength_file),
                "age_hours": None,
            }
        }

        # Calculate age
        for key, filepath in [("player_stats", self.player_stats_file),
                               ("team_strength", self.team_strength_file)]:
            if filepath.exists():
                mod_time = datetime.fromtimestamp(filepath.stat().st_mtime)
                age_hours = (datetime.now() - mod_time).total_seconds() / 3600
                info[key]["age_hours"] = round(age_hours, 1)

        return info
