#!/usr/bin/env python3
"""
Fetch External Data Script

Fetches and caches advanced stats from Understat and FBref.
Run this once per day to keep the cache fresh.

Usage:
    python scripts/fetch_external_data.py
    python scripts/fetch_external_data.py --force  # Force refresh even if cached
    python scripts/fetch_external_data.py --info   # Show cache status
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fpl_optimizer.fpl_client import FPLClient
from fpl_optimizer.models import Player, Team
from fpl_optimizer.data_sources import build_advanced_stats
from fpl_optimizer.data_cache import DataCache

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_and_cache_data(force_refresh: bool = False):
    """
    Fetch external data and cache it.

    Args:
        force_refresh: If True, ignore existing cache and re-fetch
    """
    print("\n" + "=" * 70)
    print("FPL OPTIMIZER - External Data Fetcher")
    print("=" * 70)

    # Initialize cache
    cache = DataCache()

    # Check existing cache
    cache_info = cache.get_cache_info()
    print(f"\nCache Directory: {cache_info['cache_dir']}")
    print(f"\nPlayer Stats Cache:")
    print(f"  Exists: {cache_info['player_stats']['exists']}")
    print(f"  Valid: {cache_info['player_stats']['valid']}")
    if cache_info['player_stats']['age_hours'] is not None:
        print(f"  Age: {cache_info['player_stats']['age_hours']:.1f} hours")

    print(f"\nTeam Strength Cache:")
    print(f"  Exists: {cache_info['team_strength']['exists']}")
    print(f"  Valid: {cache_info['team_strength']['valid']}")
    if cache_info['team_strength']['age_hours'] is not None:
        print(f"  Age: {cache_info['team_strength']['age_hours']:.1f} hours")

    # Check if we need to fetch
    if not force_refresh:
        player_stats_cached = cache.load_player_stats()
        team_strength_cached = cache.load_team_strength()

        if player_stats_cached is not None and team_strength_cached is not None:
            print("\n✓ Valid cache found - no need to fetch")
            print("  Use --force to refresh anyway")
            return

    print("\n" + "-" * 70)
    print("Fetching fresh data from external sources...")
    print("-" * 70)

    try:
        # Initialize FPL client
        print("\n[1/4] Fetching FPL data...")
        client = FPLClient()
        bootstrap = client.get_bootstrap_static()

        players = [Player(**p) for p in bootstrap['elements']]
        teams_data = [Team(**t) for t in bootstrap['teams']]
        teams = {t.id: t for t in teams_data}

        print(f"  ✓ Loaded {len(players)} players and {len(teams)} teams from FPL")

        # Fetch external data
        print("\n[2/4] Scraping Understat...")
        print("  (This takes 10-30 seconds due to rate limiting)")

        player_stats, team_strength = build_advanced_stats(
            players,
            teams,
            use_understat=True,  # Fetch Understat
            use_fbref=False,     # Skip FBref (slower and optional)
        )

        print(f"  ✓ Fetched stats for {len(player_stats)} players")
        print(f"  ✓ Fetched strength for {len(team_strength)} teams")

        # Cache the data
        print("\n[3/4] Caching player stats...")
        cache.save_player_stats(player_stats)
        print("  ✓ Player stats cached")

        print("\n[4/4] Caching team strength...")
        cache.save_team_strength(team_strength)
        print("  ✓ Team strength cached")

        print("\n" + "=" * 70)
        print("SUCCESS! External data fetched and cached")
        print("=" * 70)
        print("\nThe optimizer will now use advanced xG/xA data from Understat")
        print("Cache will expire in 24 hours - run this script daily for best results")
        print("\n")

    except Exception as e:
        logger.exception("Failed to fetch external data")
        print(f"\n✗ ERROR: {e}")
        print("\nFalling back to FPL data only (simple calculator)")
        sys.exit(1)


def show_cache_info():
    """Display cache information."""
    cache = DataCache()
    cache_info = cache.get_cache_info()

    print("\n" + "=" * 70)
    print("Cache Status")
    print("=" * 70)
    print(f"\nCache Directory: {cache_info['cache_dir']}")

    for name in ["player_stats", "team_strength"]:
        info = cache_info[name]
        print(f"\n{name.replace('_', ' ').title()}:")
        print(f"  Exists: {'Yes' if info['exists'] else 'No'}")
        print(f"  Valid: {'Yes' if info['valid'] else 'No (expired or missing)'}")

        if info['age_hours'] is not None:
            print(f"  Age: {info['age_hours']:.1f} hours")
            remaining = 24 - info['age_hours']
            if remaining > 0:
                print(f"  Expires in: {remaining:.1f} hours")
            else:
                print(f"  Status: EXPIRED")

    print("\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch and cache external FPL stats (Understat, FBref)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refresh even if cache is valid"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show cache status and exit"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all cached data"
    )

    args = parser.parse_args()

    if args.info:
        show_cache_info()
        return

    if args.clear:
        cache = DataCache()
        cache.clear_cache()
        print("\n✓ Cache cleared")
        return

    # Fetch and cache data
    fetch_and_cache_data(force_refresh=args.force)


if __name__ == "__main__":
    main()
