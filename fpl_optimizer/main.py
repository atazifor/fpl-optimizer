"""CLI interface for FPL optimizer."""

import sys
import argparse
from typing import List, Optional
from .fpl_client import FPLClient
from .models import Player, Team, Fixture, SquadConstraints
from .expected_points import ExpectedPointsPredictor
from .optimizer import FPLOptimizer


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70)


def print_section(text: str):
    """Print a formatted section header."""
    print("\n" + "-" * 70)
    print(text)
    print("-" * 70)


def fetch_data(client: FPLClient):
    """Fetch all necessary data from FPL API."""
    print("Fetching data from FPL API...")

    # Fetch bootstrap data
    bootstrap = client.get_bootstrap_static()

    # Parse players
    players = [Player(**p) for p in bootstrap['elements']]
    print(f"✓ Loaded {len(players)} players")

    # Parse teams
    teams_data = [Team(**t) for t in bootstrap['teams']]
    teams = {t.id: t for t in teams_data}
    print(f"✓ Loaded {len(teams)} teams")

    # Parse fixtures
    fixtures_data = client.get_fixtures()
    fixtures = [Fixture(**f) for f in fixtures_data]
    print(f"✓ Loaded {len(fixtures)} fixtures")

    # Get current gameweek
    current_gw = client.get_current_gameweek()
    print(f"✓ Current gameweek: {current_gw}")

    return players, teams, fixtures, current_gw


def cmd_optimize(args):
    """Optimize squad selection."""
    print_header("FPL SQUAD OPTIMIZER")

    # Initialize client
    client = FPLClient()

    # Fetch data
    players, teams, fixtures, current_gw = fetch_data(client)

    # Filter available players
    available_players = [p for p in players if p.is_available]
    print(f"\n✓ {len(available_players)} available players after filtering")

    # Set up constraints
    constraints = SquadConstraints(total_budget=args.budget)

    # Create optimizer
    print("\nOptimizing squad...")
    optimizer = FPLOptimizer(available_players, constraints)
    squad = optimizer.optimize_squad(objective=args.objective)

    # Display results
    print_section(f"OPTIMIZED SQUAD ({args.objective.upper()})")
    print(f"Formation: {squad.formation}")
    print(f"Total Cost: £{squad.total_cost/10:.1f}M / £{constraints.total_budget}M")
    print(f"Expected Points: {squad.expected_points:.0f}")

    # Show starting XI
    print_section("STARTING XI")
    starting = squad.get_starting_players()

    for position in ['GKP', 'DEF', 'MID', 'FWD']:
        position_players = [p for p in starting if p.position_name == position]
        if position_players:
            print(f"\n{position}:")
            for player in sorted(position_players, key=lambda p: p.total_points, reverse=True):
                captain = " (C)" if player.id == squad.captain_id else ""
                vice = " (VC)" if player.id == squad.vice_captain_id else ""
                team_name = teams[player.team_id].short_name
                print(
                    f"  {player.name:20} {team_name:3} "
                    f"£{player.cost_millions:4.1f}M  {player.total_points:3} pts{captain}{vice}"
                )

    # Show bench
    print_section("BENCH")
    bench = squad.get_bench_players()
    for i, player in enumerate(bench, 1):
        team_name = teams[player.team_id].short_name
        print(
            f"{i}. {player.position_name} - {player.name:20} {team_name:3} "
            f"£{player.cost_millions:4.1f}M  {player.total_points:3} pts"
        )


def cmd_transfers(args):
    """Recommend transfers for current squad."""
    print_header("FPL TRANSFER RECOMMENDATIONS")

    # Initialize client
    client = FPLClient()

    # Fetch data
    players, teams, fixtures, current_gw = fetch_data(client)

    # Fetch team info and history
    print("\nFetching your team details...")
    try:
        my_team = client.get_my_team()
        my_info = client.get_my_info()
        my_history = client.get_my_history()

        current_player_ids = [pick['element'] for pick in my_team['picks']]
        bank = my_team['entry_history']['bank'] / 10  # Bank in millions
        team_value = my_team['entry_history']['value'] / 10

        # Get free transfers available
        last_gw = my_history['current'][-1] if my_history['current'] else {}
        free_transfers = last_gw.get('event_transfers', 1)

        # Get chips used
        chips_used = my_history.get('chips', [])
        chips_available = {
            'wildcard': not any(c['name'] == 'wildcard' for c in chips_used),
            'freehit': not any(c['name'] == 'freehit' for c in chips_used),
            'bboost': not any(c['name'] == 'bboost' for c in chips_used),
            '3xc': not any(c['name'] == '3xc' for c in chips_used),
        }

        print(f"✓ Team: {my_info['name']}")
        print(f"✓ Team Value: £{team_value}M (£{bank}M in bank)")
        print(f"✓ Free Transfers: {free_transfers}")
        print(f"✓ Chips Available: ", end="")
        available = [k.upper() for k, v in chips_available.items() if v]
        print(", ".join(available) if available else "None")

    except Exception as e:
        print(f"Error fetching your team: {e}")
        print("Make sure you're authenticated (check .env credentials)")
        return

    # Determine optimal number of transfers
    if args.num_transfers is None:
        # Auto-determine based on free transfers
        optimal_transfers = min(free_transfers, 2)  # Don't recommend more than 2
        print(f"\n💡 Recommending {optimal_transfers} transfer(s) (you have {free_transfers} free)")
    else:
        optimal_transfers = args.num_transfers

    # Create optimizer
    print(f"\nOptimizing with {optimal_transfers} transfer(s)...")
    optimizer = FPLOptimizer(players)
    new_squad = optimizer.optimize_with_existing_players(
        current_player_ids,
        num_transfers=optimal_transfers
    )

    # Identify transfers
    current_set = set(current_player_ids)
    new_set = set(p.id for p in new_squad.players)

    players_out = current_set - new_set
    players_in = new_set - current_set
    num_transfers_made = len(players_out)

    # Calculate points cost
    transfer_cost = max(0, num_transfers_made - free_transfers) * 4

    # Get current squad points for comparison
    current_players = [p for p in players if p.id in current_player_ids]
    current_points = sum(p.total_points for p in current_players[:11])  # Assume first 11 start

    points_gain = new_squad.expected_points - current_points
    net_gain = points_gain - transfer_cost

    # Display results
    print_section(f"TRANSFER ANALYSIS")

    if not players_out:
        print("\n✓ No transfers recommended - your squad is optimal!")
    else:
        print(f"\nTransfers OUT:")
        for pid in players_out:
            player = next((p for p in players if p.id == pid), None)
            if player:
                team_name = teams[player.team_id].short_name
                print(f"  ❌ {player.name:20} {team_name:3} £{player.cost_millions:.1f}M  {player.total_points} pts")

        print(f"\nTransfers IN:")
        for pid in players_in:
            player = next((p for p in players if p.id == pid), None)
            if player:
                team_name = teams[player.team_id].short_name
                print(f"  ✅ {player.name:20} {team_name:3} £{player.cost_millions:.1f}M  {player.total_points} pts")

        print(f"\n{'='*60}")
        print(f"Transfers Made: {num_transfers_made}")
        print(f"Free Transfers: {free_transfers}")
        print(f"Transfer Cost: -{transfer_cost} points")
        print(f"Expected Points Gain: +{points_gain:.1f}")
        print(f"Net Gain: {'+' if net_gain >= 0 else ''}{net_gain:.1f} points")
        print(f"{'='*60}")

        # Recommendation
        if net_gain > 0:
            print(f"\n✅ RECOMMENDATION: Make these transfers (worth the hit!)")
        elif net_gain == 0:
            print(f"\n⚠️  RECOMMENDATION: Marginal - consider waiting")
        else:
            print(f"\n❌ RECOMMENDATION: Not worth the hit - save your transfer")

    # Show new starting XI
    print_section("RECOMMENDED STARTING XI")
    starting = new_squad.get_starting_players()

    for position in ['GKP', 'DEF', 'MID', 'FWD']:
        position_players = [p for p in starting if p.position_name == position]
        if position_players:
            print(f"\n{position}:")
            for player in sorted(position_players, key=lambda p: p.total_points, reverse=True):
                captain = " (C)" if player.id == new_squad.captain_id else ""
                vice = " (VC)" if player.id == new_squad.vice_captain_id else ""
                team_name = teams[player.team_id].short_name
                is_new = "🆕 " if player.id in players_in else "   "
                print(
                    f"  {is_new}{player.name:20} {team_name:3} "
                    f"£{player.cost_millions:4.1f}M{captain}{vice}"
                )


def cmd_captain(args):
    """Recommend captain choice."""
    print_header("FPL CAPTAIN RECOMMENDATION")

    # Initialize client
    client = FPLClient()

    # Fetch data
    players, teams, fixtures, current_gw = fetch_data(client)

    # Fetch current team
    print("\nFetching your current team...")
    try:
        my_team = client.get_my_team()
        current_player_ids = [pick['element'] for pick in my_team['picks']]
        squad_players = [p for p in players if p.id in current_player_ids]
        print(f"✓ Analyzing {len(squad_players)} players in your squad")
    except Exception as e:
        print(f"Error fetching your team: {e}")
        print("Make sure FPL_EMAIL and FPL_PASSWORD are set in .env file")
        return

    # Create predictor
    predictor = ExpectedPointsPredictor(players, teams, fixtures)

    # Calculate captain scores
    print("\nCalculating expected points for each player...")
    captain_scores = []

    for player in squad_players[:11]:  # Only consider starting 11
        expected_pts = predictor.calculate_expected_points(player, current_gw, num_gameweeks=1)
        captain_value = expected_pts * 2  # Captain gets double points
        captain_scores.append((player, expected_pts, captain_value))

    # Sort by captain value
    captain_scores.sort(key=lambda x: x[2], reverse=True)

    # Display top 5 captain options
    print_section("TOP 5 CAPTAIN OPTIONS")

    for i, (player, exp_pts, captain_pts) in enumerate(captain_scores[:5], 1):
        team_name = teams[player.team_id].short_name
        fixtures_str = ""

        # Show next fixture
        player_fixtures = [
            f for f in fixtures
            if (f.team_h == player.team_id or f.team_a == player.team_id)
            and f.event == current_gw
        ]

        if player_fixtures:
            fixture = player_fixtures[0]
            if fixture.team_h == player.team_id:
                opponent = teams[fixture.team_a].short_name
                fixtures_str = f"vs {opponent} (H)"
            else:
                opponent = teams[fixture.team_h].short_name
                fixtures_str = f"@ {opponent} (A)"

        emoji = "⭐" if i == 1 else f"{i}."
        print(
            f"{emoji} {player.name:20} {team_name:3} "
            f"£{player.cost_millions:4.1f}M  {fixtures_str:15}  "
            f"Expected: {captain_pts:.1f} pts"
        )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FPL Optimizer - Optimize your Fantasy Premier League team",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize squad selection')
    optimize_parser.add_argument(
        '--budget',
        type=float,
        default=100.0,
        help='Total budget in millions (default: 100.0)'
    )
    optimize_parser.add_argument(
        '--objective',
        choices=['points', 'value', 'form', 'expected'],
        default='points',
        help='Optimization objective (default: points)'
    )

    # Transfers command
    transfers_parser = subparsers.add_parser('transfers', help='Recommend transfers')
    transfers_parser.add_argument(
        '--num-transfers',
        type=int,
        default=None,
        help='Number of transfers to make (default: auto-detect from free transfers)'
    )

    # Captain command
    captain_parser = subparsers.add_parser('captain', help='Recommend captain choice')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == 'optimize':
            cmd_optimize(args)
        elif args.command == 'transfers':
            cmd_transfers(args)
        elif args.command == 'captain':
            cmd_captain(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()