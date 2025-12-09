#!/usr/bin/env python3
"""
Fetch Historical Fixtures

Fetches fixture data for each gameweek and caches it.
This is needed for realistic backtesting with the expected points calculator.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import requests
import time
from pathlib import Path

def fetch_fixtures_for_gameweek(gw: int, cache_dir: Path) -> list:
    """Fetch fixtures for a specific gameweek."""
    cache_file = cache_dir / f"fixtures_gw{gw}.json"

    # Check cache
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    # Fetch from API
    url = f"https://fantasy.premierleague.com/api/fixtures/?event={gw}"
    print(f"  Fetching GW{gw} fixtures...")

    try:
        response = requests.get(url)
        response.raise_for_status()
        fixtures = response.json()

        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(fixtures, f, indent=2)

        time.sleep(0.5)  # Rate limiting
        return fixtures

    except Exception as e:
        print(f"  Error fetching GW{gw}: {e}")
        return []


def main():
    """Fetch and cache all fixtures."""
    cache_dir = Path("backtest_cache/fixtures")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("FETCHING HISTORICAL FIXTURES")
    print("="*80)
    print("\nFetching fixture data for GW1-15...")
    print("This is needed for realistic backtesting.\n")

    all_fixtures = {}

    for gw in range(1, 16):  # GW1-15
        fixtures = fetch_fixtures_for_gameweek(gw, cache_dir)

        if fixtures:
            # Filter to only finished fixtures
            finished = [f for f in fixtures if f.get('finished', False)]
            all_fixtures[gw] = finished
            print(f"  ✓ GW{gw}: {len(finished)} finished fixtures")
        else:
            print(f"  ✗ GW{gw}: No data")
            break

    # Save combined file
    combined_file = cache_dir / "all_fixtures.json"
    with open(combined_file, 'w') as f:
        json.dump(all_fixtures, f, indent=2)

    print(f"\n✓ Cached {len(all_fixtures)} gameweeks")
    print(f"✓ Saved to: {cache_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()