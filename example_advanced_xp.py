"""
Example: Using the Advanced Expected Points Calculator

This demonstrates:
1. Basic usage with FPL data only (Simple calculator)
2. Advanced usage with Understat data (Advanced calculator)
3. How the optimizer uses the improved xP predictions
"""

import logging
from fpl_optimizer.fpl_client import FPLClient
from fpl_optimizer.api import FPLDataService
from fpl_optimizer.models import Player, Team, Fixture
from fpl_optimizer.expected_points import (
    create_calculator,
    SimpleExpectedPointsCalculator,
    AdvancedExpectedPointsCalculator,
)
from fpl_optimizer.data_sources import build_advanced_stats
from fpl_optimizer.optimizer import FPLOptimizer, OptimizationConfig, ObjectiveType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_simple_calculator():
    """Example 1: Using the simple calculator (FPL data only)."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Expected Points Calculator (FPL Data Only)")
    print("="*70)

    # Initialize FPL client and data service
    client = FPLClient()
    data_service = FPLDataService(client)

    # Load data
    players = data_service.get_players()
    teams = data_service.get_teams()
    fixtures = data_service.get_fixtures()

    # Create simple calculator
    calculator = SimpleExpectedPointsCalculator()

    # Get current gameweek
    current_gw = client.get_current_gameweek()

    # Calculate xP for top players
    print(f"\nCalculating xP for Gameweek {current_gw}\n")

    # Find some premium players
    premium_players = [p for p in players if p.cost_millions >= 9.0][:10]

    print(f"{'Player':<25} {'Team':<15} {'Cost':<8} {'xP':<8} {'Form':<8}")
    print("-" * 70)

    for player in premium_players:
        breakdown = calculator.calculate(player, current_gw, fixtures, teams)
        print(f"{player.name:<25} {teams[player.team_id].short_name:<15} "
              f"£{player.cost_millions:<7.1f} {breakdown.total_expected_points:<8.2f} {player.form:<8}")

    print("\n✓ Simple calculator uses form and fixture difficulty")
    print("✓ Good baseline but limited by FPL data only")


def example_2_advanced_calculator():
    """Example 2: Using the advanced calculator with Understat data."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Advanced Expected Points Calculator (with Understat)")
    print("="*70)

    # Initialize FPL client and data service
    client = FPLClient()
    data_service = FPLDataService(client)

    # Load FPL data
    players = data_service.get_players()
    teams = data_service.get_teams()
    fixtures = data_service.get_fixtures()

    print("\nFetching advanced stats from Understat...")
    print("(This may take 10-30 seconds due to rate limiting)\n")

    # Build advanced stats (scrapes Understat)
    try:
        player_stats, team_strength = build_advanced_stats(
            players,
            teams,
            use_understat=True,  # Scrape Understat for xG/xA
            use_fbref=False,     # Skip FBref (slower)
        )

        # Create advanced calculator with external data
        calculator = AdvancedExpectedPointsCalculator(player_stats, team_strength)

        # Get current gameweek
        current_gw = client.get_current_gameweek()

        # Calculate xP for top players with detailed breakdown
        print(f"Calculating ADVANCED xP for Gameweek {current_gw}\n")

        premium_players = [p for p in players if p.cost_millions >= 10.0][:5]

        for player in premium_players:
            breakdown = calculator.calculate(player, current_gw, fixtures, teams)

            print(f"\n{player.name} (£{player.cost_millions}M) - {teams[player.team_id].name}")
            print("-" * 70)
            print(f"  Total xP:          {breakdown.total_expected_points:.2f} points")
            print(f"  Expected Minutes:  {breakdown.minutes_expected:.0f} mins")
            print(f"  Appearance Points: {breakdown.appearance_points:.2f}")
            print(f"  Goal xP:          {breakdown.goal_expected:.3f} goals → {breakdown.goal_points:.2f} pts")
            print(f"  Assist xP:        {breakdown.assist_expected:.3f} assists → {breakdown.assist_points:.2f} pts")
            print(f"  Clean Sheet Prob: {breakdown.clean_sheet_prob:.1%} → {breakdown.clean_sheet_points:.2f} pts")
            print(f"  Bonus Expected:   {breakdown.bonus_expected:.2f} pts (BPS: {breakdown.expected_bps:.0f})")
            print(f"  Confidence:       {breakdown.confidence:.1%}")
            print(f"  Ceiling (90%):    {breakdown.ceiling:.1f} pts")
            print(f"  Floor (10%):      {breakdown.floor:.1f} pts")

            if breakdown.is_double_gameweek:
                print(f"  ⚠️  DOUBLE GAMEWEEK - {len(breakdown.fixtures)} fixtures")

        print("\n✓ Advanced calculator uses xG, xA, shot data, BPS modeling")
        print("✓ Accounts for penalties, form weighting, fixture congestion")
        print("✓ Provides ceiling/floor for risk assessment")

    except Exception as e:
        logger.error(f"Failed to fetch Understat data: {e}")
        print("\n✗ Could not fetch Understat data")
        print("  Falling back to simple calculator...")


def example_3_optimizer_integration():
    """Example 3: How the optimizer uses the improved xP calculator."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Optimizer with Advanced xP Calculator")
    print("="*70)

    # Initialize FPL client and data service
    client = FPLClient()
    data_service = FPLDataService(client)

    # Load data
    players = data_service.get_players()
    teams = data_service.get_teams()
    fixtures = data_service.get_fixtures()

    # Option A: Use simple calculator (fast, FPL data only)
    print("\nOption A: Optimizing with Simple Calculator...")
    simple_calc = SimpleExpectedPointsCalculator()
    optimizer_simple = FPLOptimizer(players, fixtures, teams, xp_calculator=simple_calc)

    config = OptimizationConfig(
        objective=ObjectiveType.POINTS,
        horizon_weeks=1,
    )

    squad_simple = optimizer_simple.optimize_squad(config)
    print(f"\n✓ Optimized Squad (Simple xP)")
    print(f"  Formation: {squad_simple.formation}")
    print(f"  Total Cost: £{squad_simple.total_cost / 10:.1f}M")
    print(f"  Expected Points: {squad_simple.expected_points:.1f}")

    # Option B: Use advanced calculator (slower, but more accurate)
    print("\n\nOption B: Optimizing with Advanced Calculator...")
    print("(Fetching Understat data... this takes 10-30 seconds)\n")

    try:
        player_stats, team_strength = build_advanced_stats(
            players, teams, use_understat=True, use_fbref=False
        )

        advanced_calc = AdvancedExpectedPointsCalculator(player_stats, team_strength)
        optimizer_advanced = FPLOptimizer(players, fixtures, teams, xp_calculator=advanced_calc)

        squad_advanced = optimizer_advanced.optimize_squad(config)

        print(f"\n✓ Optimized Squad (Advanced xP)")
        print(f"  Formation: {squad_advanced.formation}")
        print(f"  Total Cost: £{squad_advanced.total_cost / 10:.1f}M")
        print(f"  Expected Points: {squad_advanced.expected_points:.1f}")

        print(f"\n📊 Comparison:")
        print(f"  xP Difference: {squad_advanced.expected_points - squad_simple.expected_points:+.1f} points")
        print(f"  The advanced model finds {len(set(squad_advanced.starting_11_ids) - set(squad_simple.starting_11_ids))} different players")

    except Exception as e:
        logger.error(f"Failed to use advanced calculator: {e}")
        print("\n✗ Could not use advanced calculator")


def example_4_differential_hunting():
    """Example 4: Using differential_value for finding low-owned gems."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Differential Hunting (Low Ownership + High xP)")
    print("="*70)

    # Initialize FPL client and data service
    client = FPLClient()
    data_service = FPLDataService(client)

    # Load data
    players = data_service.get_players()
    teams = data_service.get_teams()
    fixtures = data_service.get_fixtures()

    # Use simple calculator for speed
    calculator = SimpleExpectedPointsCalculator()
    current_gw = client.get_current_gameweek()

    # Find differentials (low ownership, good xP)
    print(f"\nFinding differentials for Gameweek {current_gw}...")
    print("(Players with <10% ownership and xP > 4.0)\n")

    differentials = []

    for player in players:
        if player.selected_by_percent < 10.0 and player.cost_millions >= 5.0:
            breakdown = calculator.calculate(player, current_gw, fixtures, teams)
            if breakdown.total_expected_points >= 4.0:
                differentials.append((player, breakdown))

    # Sort by differential value
    differentials.sort(key=lambda x: x[1].differential_value, reverse=True)

    print(f"{'Player':<25} {'Team':<12} {'Cost':<8} {'Own%':<8} {'xP':<8} {'Diff Value':<12}")
    print("-" * 90)

    for player, breakdown in differentials[:15]:
        print(f"{player.name:<25} {teams[player.team_id].short_name:<12} "
              f"£{player.cost_millions:<7.1f} {player.selected_by_percent:<7.1f}% "
              f"{breakdown.total_expected_points:<7.2f} {breakdown.differential_value:<12.2f}")

    print("\n✓ Differential value = xP adjusted for ownership")
    print("✓ Low-owned players with high xP can help you gain ranks")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("FPL OPTIMIZER - Advanced Expected Points System")
    print("="*70)

    try:
        # Example 1: Simple calculator (fast)
        example_1_simple_calculator()

        # Example 2: Advanced calculator with Understat (slow but accurate)
        # Uncomment to run (takes 30+ seconds due to scraping):
        # example_2_advanced_calculator()

        # Example 3: Optimizer integration
        example_3_optimizer_integration()

        # Example 4: Differential hunting
        example_4_differential_hunting()

        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70 + "\n")

    except Exception as e:
        logger.exception(f"Example failed: {e}")
        print(f"\n✗ Example failed: {e}")


if __name__ == "__main__":
    main()