"""
Simple validation test for the advanced xP system.

This runs basic smoke tests to ensure the system works.
"""

import sys
from fpl_optimizer.fpl_client import FPLClient
from fpl_optimizer.models import Player, Team, Fixture
from fpl_optimizer.expected_points import (
    SimpleExpectedPointsCalculator,
    AdvancedExpectedPointsCalculator,
    PlayerStats,
    TeamStrength,
)
from fpl_optimizer.optimizer import FPLOptimizer, OptimizationConfig


def test_simple_calculator():
    """Test that simple calculator works with FPL data."""
    print("\n[TEST 1] Testing Simple Calculator...")

    try:
        # Load data
        client = FPLClient()
        bootstrap = client.get_bootstrap_static()

        players = [Player(**p) for p in bootstrap['elements'][:50]]  # Test with first 50
        teams = {t['id']: Team(**t) for t in bootstrap['teams']}
        fixtures = [Fixture(**f) for f in client.get_fixtures()[:20]]  # Test with first 20

        # Create calculator
        calculator = SimpleExpectedPointsCalculator()

        # Calculate xP for first available player
        current_gw = client.get_current_gameweek()
        test_player = next((p for p in players if p.is_available and p.cost_millions >= 5.0), None)

        if not test_player:
            print("  ✗ No available test player found")
            return False

        breakdown = calculator.calculate(test_player, current_gw, fixtures, teams)

        # Validate breakdown
        assert hasattr(breakdown, 'total_expected_points'), "Missing total_expected_points"
        assert breakdown.total_expected_points >= 0, "Negative xP"
        assert breakdown.total_expected_points < 30, "Unrealistic xP"
        assert breakdown.confidence >= 0 and breakdown.confidence <= 1, "Invalid confidence"

        print(f"  ✓ Simple calculator works")
        print(f"    Player: {test_player.name}")
        print(f"    xP: {breakdown.total_expected_points:.2f}")
        print(f"    Confidence: {breakdown.confidence:.1%}")
        return True

    except Exception as e:
        print(f"  ✗ Simple calculator failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advanced_calculator_structure():
    """Test that advanced calculator works (without external data)."""
    print("\n[TEST 2] Testing Advanced Calculator Structure...")

    try:
        # Load data
        client = FPLClient()
        bootstrap = client.get_bootstrap_static()

        players = [Player(**p) for p in bootstrap['elements'][:50]]
        teams = {t['id']: Team(**t) for t in bootstrap['teams']}
        fixtures = [Fixture(**f) for f in client.get_fixtures()[:20]]

        # Create mock advanced stats for one player
        test_player = next((p for p in players if p.is_available and p.cost_millions >= 8.0), None)

        if not test_player:
            print("  ✗ No test player found")
            return False

        player_stats = {
            test_player.id: PlayerStats(
                player_id=test_player.id,
                position=test_player.position_name,
                team_id=test_player.team_id,
                xg_per_90=0.45,
                xa_per_90=0.25,
                xgi_per_90=0.70,
                shots_per_90=3.2,
                shots_on_target_per_90=1.5,
                key_passes_per_90=2.1,
                avg_minutes_per_game=85,
                total_minutes=test_player.minutes,
                games_started=test_player.minutes // 85,
                is_penalty_taker=True,
                penalty_order=1,
                penalties_taken=3,
                penalties_scored=2,
                form_xgi_per_90=0.80,
                form_games=5,
            )
        }

        team_strength = {
            test_player.team_id: TeamStrength(
                team_id=test_player.team_id,
                team_name=teams[test_player.team_id].name,
                xg_per_90=1.8,
                xga_per_90=1.1,
                home_xg_per_90=2.0,
                away_xg_per_90=1.6,
                home_xga_per_90=1.0,
                away_xga_per_90=1.2,
                clean_sheet_percentage=35.0,
                penalties_won_per_90=0.12,
            )
        }

        # Create advanced calculator
        calculator = AdvancedExpectedPointsCalculator(player_stats, team_strength)

        # Calculate xP
        current_gw = client.get_current_gameweek()
        breakdown = calculator.calculate(test_player, current_gw, fixtures, teams)

        # Validate breakdown has all components
        assert hasattr(breakdown, 'total_expected_points'), "Missing total_expected_points"
        assert hasattr(breakdown, 'goal_expected'), "Missing goal_expected"
        assert hasattr(breakdown, 'assist_expected'), "Missing assist_expected"
        assert hasattr(breakdown, 'bonus_expected'), "Missing bonus_expected"
        assert hasattr(breakdown, 'expected_bps'), "Missing expected_bps"
        assert hasattr(breakdown, 'ceiling'), "Missing ceiling"
        assert hasattr(breakdown, 'floor'), "Missing floor"
        assert hasattr(breakdown, 'differential_value'), "Missing differential_value"

        # Validate values are reasonable
        assert breakdown.total_expected_points >= 0, "Negative xP"
        assert breakdown.total_expected_points < 30, "Unrealistic xP"
        assert breakdown.ceiling >= breakdown.total_expected_points, "Ceiling < xP"
        assert breakdown.floor <= breakdown.total_expected_points, "Floor > xP"

        print(f"  ✓ Advanced calculator structure valid")
        print(f"    Player: {test_player.name}")
        print(f"    xP: {breakdown.total_expected_points:.2f}")
        print(f"    Goal xP: {breakdown.goal_expected:.3f}")
        print(f"    Assist xP: {breakdown.assist_expected:.3f}")
        print(f"    Bonus: {breakdown.bonus_expected:.2f}")
        print(f"    Ceiling: {breakdown.ceiling:.1f} | Floor: {breakdown.floor:.1f}")
        return True

    except Exception as e:
        print(f"  ✗ Advanced calculator failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimizer_integration():
    """Test that optimizer works with new calculator."""
    print("\n[TEST 3] Testing Optimizer Integration...")

    try:
        # Load data
        client = FPLClient()
        bootstrap = client.get_bootstrap_static()

        players = [Player(**p) for p in bootstrap['elements']]
        teams = {t['id']: Team(**t) for t in bootstrap['teams']}
        fixtures = [Fixture(**f) for f in client.get_fixtures()]

        # Create simple calculator
        calculator = SimpleExpectedPointsCalculator()

        # Create optimizer
        optimizer = FPLOptimizer(players, fixtures, teams, xp_calculator=calculator)

        # Run optimization
        config = OptimizationConfig(horizon_weeks=1)
        squad = optimizer.optimize_squad(config)

        # Validate result
        assert squad is not None, "No squad returned"
        assert len(squad.players) == 15, f"Wrong squad size: {len(squad.players)}"
        assert len(squad.starting_11_ids) == 11, f"Wrong starting 11 size: {len(squad.starting_11_ids)}"
        assert squad.total_cost <= 1000, f"Over budget: {squad.total_cost}"
        assert squad.expected_points > 0, "No expected points"

        print(f"  ✓ Optimizer integration works")
        print(f"    Formation: {squad.formation}")
        print(f"    Cost: £{squad.total_cost / 10:.1f}M")
        print(f"    Expected Points: {squad.expected_points:.1f}")
        return True

    except Exception as e:
        print(f"  ✗ Optimizer integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("ADVANCED XP SYSTEM - VALIDATION TESTS")
    print("="*70)

    results = []

    # Run tests
    results.append(("Simple Calculator", test_simple_calculator()))
    results.append(("Advanced Calculator", test_advanced_calculator_structure()))
    results.append(("Optimizer Integration", test_optimizer_integration()))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status} - {test_name}")

    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! The advanced xP system is working.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())