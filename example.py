#!/usr/bin/env python3
"""Example script demonstrating FPL optimizer usage."""

from fpl_optimizer.api import FPLDataFetcher
from fpl_optimizer.models import Player, SquadConstraints
from fpl_optimizer.optimizer import FPLOptimizer


def main():
    """Run the FPL optimizer example."""
    print("=" * 60)
    print("FPL Team Optimizer - Example")
    print("=" * 60)

    # 1. Fetch data from FPL API
    print("\n[1/4] Fetching data from FPL API...")
    fetcher = FPLDataFetcher()
    fetcher.fetch_bootstrap_static()
    print(f"✓ Fetched data for {len(fetcher.bootstrap_data['elements'])} players")

    # 2. Convert to Player models
    print("\n[2/4] Converting to data models...")
    players = [
        Player.from_api_data(p)
        for p in fetcher.bootstrap_data['elements']
    ]
    print(f"✓ Converted {len(players)} players to model objects")

    # 3. Set up constraints
    print("\n[3/4] Setting up constraints...")
    constraints = SquadConstraints(
        total_budget=100.0,
        max_players_per_team=3
    )
    print(f"✓ Budget: £{constraints.total_budget}M")
    print(f"✓ Max players per team: {constraints.max_players_per_team}")

    # 4. Optimize squad
    print("\n[4/4] Optimizing squad...")
    optimizer = FPLOptimizer(players, constraints)
    squad = optimizer.optimize_squad(objective='points')
    print("✓ Optimization complete!")

    # Display results
    print("\n" + "=" * 60)
    print("OPTIMIZED SQUAD")
    print("=" * 60)
    print(f"\nFormation: {squad.formation}")
    print(f"Total Cost: £{squad.total_cost:.1f}M / £{constraints.total_budget}M")
    print(f"Expected Points: {squad.expected_points:.0f}")

    # Starting XI
    print("\n" + "-" * 60)
    print("STARTING XI")
    print("-" * 60)
    starting = squad.get_starting_players()

    # Group by position
    for position in ['GKP', 'DEF', 'MID', 'FWD']:
        position_players = [p for p in starting if p.position == position]
        if position_players:
            print(f"\n{position}:")
            for player in sorted(position_players, key=lambda p: p.total_points, reverse=True):
                captain = " (C)" if player.id == squad.captain_id else ""
                vice = " (VC)" if player.id == squad.vice_captain_id else ""
                print(
                    f"  {player.name:20} £{player.cost:4.1f}M  "
                    f"{player.total_points:3.0f} pts{captain}{vice}"
                )

    # Bench
    print("\n" + "-" * 60)
    print("BENCH")
    print("-" * 60)
    bench = squad.get_bench_players()
    for i, player in enumerate(bench, 1):
        print(
            f"{i}. {player.position:3} - {player.name:20} £{player.cost:4.1f}M  "
            f"{player.total_points:3.0f} pts"
        )

    # Squad summary by team
    print("\n" + "-" * 60)
    print("PLAYERS BY TEAM")
    print("-" * 60)
    teams_dict = {t['id']: t['short_name'] for t in fetcher.bootstrap_data['teams']}
    team_counts = {}
    for player in squad.players:
        team_name = teams_dict.get(player.team_id, 'Unknown')
        if team_name not in team_counts:
            team_counts[team_name] = []
        team_counts[team_name].append(player)

    for team_name in sorted(team_counts.keys()):
        players_list = team_counts[team_name]
        print(f"\n{team_name} ({len(players_list)}):")
        for player in players_list:
            print(f"  {player.position} - {player.name}")

    print("\n" + "=" * 60)

    # Try different optimization objectives
    print("\n\nTrying different optimization objectives...")
    print("-" * 60)

    objectives = {
        'value': 'Points per Million',
        'form': 'Current Form',
        'expected': 'Expected Goal Involvements'
    }

    for obj_key, obj_name in objectives.items():
        squad_alt = optimizer.optimize_squad(objective=obj_key)
        print(f"\n{obj_name}:")
        print(f"  Formation: {squad_alt.formation}")
        print(f"  Cost: £{squad_alt.total_cost:.1f}M")
        print(f"  Expected Points: {squad_alt.expected_points:.0f}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()