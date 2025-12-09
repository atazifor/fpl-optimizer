"""
Advanced Expected Points Calculation Engine.

This module provides sophisticated expected points predictions using:
- Underlying statistics (xG, xA, shots, key passes)
- Position-specific modeling
- Opponent strength analysis
- Minutes and rotation prediction
- Bonus point probability from BPS modeling
- Clean sheet probability
- Set piece and penalty consideration
- Form weighting and fixture congestion

Data sources to integrate:
- FPL API (baseline stats)
- Understat (xG, xA, advanced metrics)
- FBref (detailed passing, defensive stats)
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .models import Player, Fixture, Team

logger = logging.getLogger(__name__)


# =============================================================================
# Points System Constants
# =============================================================================

@dataclass(frozen=True)
class PointsSystem:
    """FPL points awarded for each action by position."""

    # Appearance points
    MINUTES_1_TO_59: int = 1
    MINUTES_60_PLUS: int = 2

    # Goals by position
    GOAL_GKP: int = 6
    GOAL_DEF: int = 6
    GOAL_MID: int = 5
    GOAL_FWD: int = 4

    # Assists (all positions)
    ASSIST: int = 3

    # Clean sheets
    CLEAN_SHEET_GKP: int = 4
    CLEAN_SHEET_DEF: int = 4
    CLEAN_SHEET_MID: int = 1
    CLEAN_SHEET_FWD: int = 0

    # Saves (GKP only)
    SAVES_PER_POINT: int = 3  # 1 point per 3 saves

    # Penalties
    PENALTY_SAVE: int = 5
    PENALTY_MISS: int = -2

    # Bonus points
    MAX_BONUS: int = 3

    # Negative
    GOAL_CONCEDED_PER_TWO: int = -1  # -1 per 2 goals for GKP/DEF
    YELLOW_CARD: int = -1
    RED_CARD: int = -3
    OWN_GOAL: int = -2

    # BPS (Bonus Point System) values
    BPS_GOAL_FWD: int = 24
    BPS_GOAL_MID: int = 30
    BPS_GOAL_DEF: int = 36
    BPS_GOAL_GKP: int = 42
    BPS_ASSIST: int = 18
    BPS_CLEAN_SHEET_DEF_GKP: int = 12
    BPS_SHOT_ON_TARGET: int = 2
    BPS_KEY_PASS: int = 1
    BPS_SAVE: int = 2  # Per 3 saves
    BPS_PENALTY_SAVE: int = 15
    BPS_TACKLE_WON: int = 2
    BPS_INTERCEPTION: int = 3
    BPS_CLEARANCE_BLOCK_INTERCEPTION: int = 1
    BPS_GOALS_CONCEDED_PER_TWO: int = -4  # For GKP/DEF

    def goal_points(self, position: str) -> int:
        """Get goal points for position."""
        return {
            'GKP': self.GOAL_GKP,
            'DEF': self.GOAL_DEF,
            'MID': self.GOAL_MID,
            'FWD': self.GOAL_FWD,
        }.get(position, self.GOAL_FWD)

    def clean_sheet_points(self, position: str) -> int:
        """Get clean sheet points for position."""
        return {
            'GKP': self.CLEAN_SHEET_GKP,
            'DEF': self.CLEAN_SHEET_DEF,
            'MID': self.CLEAN_SHEET_MID,
            'FWD': self.CLEAN_SHEET_FWD,
        }.get(position, 0)

    def bps_per_goal(self, position: str) -> int:
        """Get BPS awarded per goal by position."""
        return {
            'GKP': self.BPS_GOAL_GKP,
            'DEF': self.BPS_GOAL_DEF,
            'MID': self.BPS_GOAL_MID,
            'FWD': self.BPS_GOAL_FWD,
        }.get(position, self.BPS_GOAL_FWD)


POINTS = PointsSystem()


# =============================================================================
# Data Models for Predictions
# =============================================================================

@dataclass
class PlayerStats:
    """Extended player statistics for prediction."""

    # Core identifiers
    player_id: int
    position: str
    team_id: int

    # Expected stats (per 90 minutes) - from Understat/FBref
    xg_per_90: float = 0.0          # Expected goals
    xa_per_90: float = 0.0          # Expected assists
    xgi_per_90: float = 0.0         # Expected goal involvement (xG + xA)

    # Actual underlying stats (per 90) - from FBref
    shots_per_90: float = 0.0
    shots_on_target_per_90: float = 0.0
    key_passes_per_90: float = 0.0
    big_chances_created_per_90: float = 0.0

    # Defensive stats (for DEF/GKP)
    clean_sheet_percentage: float = 0.0
    saves_per_90: float = 0.0       # GKP only
    tackles_per_90: float = 0.0     # DEF
    interceptions_per_90: float = 0.0  # DEF
    clearances_per_90: float = 0.0  # DEF

    # Set pieces
    is_penalty_taker: bool = False
    penalty_order: int = 0          # 1 = first choice, 2 = backup
    penalties_taken: int = 0
    penalties_scored: int = 0
    is_set_piece_taker: bool = False
    corners_per_90: float = 0.0
    direct_free_kicks_per_90: float = 0.0

    # Minutes and availability
    avg_minutes_per_game: float = 0.0
    games_started: int = 0
    games_subbed_on: int = 0
    games_subbed_off: int = 0
    total_minutes: int = 0

    # Bonus points
    avg_bonus_per_game: float = 0.0
    avg_bps_per_90: float = 0.0     # BPS = Bonus Point System raw score

    # Form (rolling averages - last 5 games)
    form_xgi_per_90: float = 0.0    # Recent xGI
    form_minutes: float = 0.0       # Avg minutes last 5
    form_games: int = 0             # Number of games in form calculation

    # Consistency
    points_variance: float = 0.0     # How volatile are returns?
    blank_percentage: float = 0.0    # % of games with ≤2 points


@dataclass
class TeamStrength:
    """Team attacking and defensive strength metrics."""

    team_id: int
    team_name: str = ""

    # Offensive (higher = better attack)
    xg_per_90: float = 0.0
    goals_per_90: float = 0.0
    shots_per_90: float = 0.0

    # Defensive (lower = better defense)
    xga_per_90: float = 0.0         # Expected goals against
    goals_conceded_per_90: float = 0.0
    shots_conceded_per_90: float = 0.0

    # Home/away splits
    home_xg_per_90: float = 0.0
    home_xga_per_90: float = 0.0
    away_xg_per_90: float = 0.0
    away_xga_per_90: float = 0.0

    # Clean sheet probability (from actual data)
    clean_sheet_percentage: float = 0.0
    home_cs_percentage: float = 0.0
    away_cs_percentage: float = 0.0

    # Set pieces
    penalties_won_per_90: float = 0.0
    penalties_conceded_per_90: float = 0.0


@dataclass
class FixturePrediction:
    """Predicted match outcome for xP calculation."""

    fixture_id: int
    gameweek: int

    # Team perspective
    team_id: int
    opponent_id: int
    is_home: bool

    # Expected goals
    team_xg: float = 0.0            # How many goals team expected to score
    opponent_xg: float = 0.0        # How many opponent expected to score

    # Derived probabilities
    clean_sheet_prob: float = 0.0
    win_prob: float = 0.0

    # Difficulty (1-5 scale, but data-driven)
    calculated_difficulty: float = 3.0


@dataclass
class ExpectedPointsBreakdown:
    """Detailed breakdown of expected points."""

    player_id: int
    gameweek: int

    # Component predictions
    minutes_expected: float = 0.0
    appearance_points: float = 0.0

    goal_expected: float = 0.0
    goal_points: float = 0.0

    assist_expected: float = 0.0
    assist_points: float = 0.0

    clean_sheet_prob: float = 0.0
    clean_sheet_points: float = 0.0

    bonus_expected: float = 0.0
    expected_bps: float = 0.0       # Raw BPS score

    save_points: float = 0.0        # GKP only

    # Negative expectations
    goals_conceded_penalty: float = 0.0
    card_penalty: float = 0.0

    # Total
    total_expected_points: float = 0.0

    # Risk metrics
    confidence: float = 0.0         # 0-1, based on sample size
    variance: float = 0.0           # Expected variance in outcome
    ceiling: float = 0.0            # 90th percentile outcome
    floor: float = 0.0              # 10th percentile outcome

    # Differential value (for optimization)
    ownership: float = 0.0
    differential_value: float = 0.0  # Adjusted for low ownership

    # Metadata
    fixtures: List[int] = field(default_factory=list)
    is_double_gameweek: bool = False


# =============================================================================
# Expected Points Calculators
# =============================================================================

class ExpectedPointsCalculator(ABC):
    """Abstract base class for expected points models."""

    @abstractmethod
    def calculate(
        self,
        player: Player,
        gameweek: int,
        fixtures: List[Fixture],
        teams: Dict[int, Team],
    ) -> ExpectedPointsBreakdown:
        """Calculate expected points with full breakdown."""
        pass

    def get_total(
        self,
        player: Player,
        gameweek: int,
        fixtures: List[Fixture],
        teams: Dict[int, Team],
    ) -> float:
        """Get just the total expected points."""
        return self.calculate(player, gameweek, fixtures, teams).total_expected_points


class AdvancedExpectedPointsCalculator(ExpectedPointsCalculator):
    """
    Production-grade expected points calculator.

    Uses underlying statistics to predict:
    1. Minutes probability (with fixture congestion)
    2. Goals (from xG, shots, opponent defense, penalties)
    3. Assists (from xA, key passes, team attack)
    4. Clean sheets (from team defense, opponent attack)
    5. Bonus points (from BPS modeling)
    6. Save points (GKP)
    7. Card probability
    8. Form weighting
    """

    def __init__(
        self,
        player_stats: Optional[Dict[int, PlayerStats]] = None,
        team_strength: Optional[Dict[int, TeamStrength]] = None,
        league_avg_goals_per_game: float = 2.8,  # EPL average
        league_avg_pens_per_game: float = 0.08,  # EPL average
    ):
        """
        Initialize with optional advanced stats.

        Args:
            player_stats: Advanced stats by player ID (from Understat/FBref)
            team_strength: Team strength metrics by team ID
            league_avg_goals_per_game: League average for normalization
            league_avg_pens_per_game: League average penalties per game
        """
        self.player_stats = player_stats or {}
        self.team_strength = team_strength or {}
        self.league_avg = league_avg_goals_per_game
        self.league_avg_pens = league_avg_pens_per_game

        # Cache for fixture predictions
        self._fixture_cache: Dict[Tuple[int, int, int], FixturePrediction] = {}

    def calculate(
        self,
        player: Player,
        gameweek: int,
        fixtures: List[Fixture],
        teams: Dict[int, Team],
    ) -> ExpectedPointsBreakdown:
        """Calculate expected points with full breakdown."""

        # Get player's fixtures for this gameweek
        player_fixtures = [
            f for f in fixtures
            if f.event == gameweek and (f.team_h == player.team_id or f.team_a == player.team_id)
        ]

        if not player_fixtures:
            return ExpectedPointsBreakdown(
                player_id=player.id,
                gameweek=gameweek,
                confidence=0.0,
            )

        # Check availability
        if not self._is_available(player):
            return ExpectedPointsBreakdown(
                player_id=player.id,
                gameweek=gameweek,
                confidence=1.0,  # Confident they won't play
            )

        # Get advanced stats if available
        stats = self.player_stats.get(player.id)

        # Get fixture congestion adjustment
        congestion_mult = self._get_congestion_adjustment(player, gameweek, fixtures)

        # Calculate each component
        breakdown = ExpectedPointsBreakdown(
            player_id=player.id,
            gameweek=gameweek,
            fixtures=[f.id for f in player_fixtures],
            is_double_gameweek=len(player_fixtures) > 1,
            ownership=player.selected_by_percent,
        )

        # Accumulate across fixtures (handles DGW)
        for fixture in player_fixtures:
            fixture_pred = self._predict_fixture(fixture, player.team_id, teams)

            # Minutes (with congestion adjustment)
            mins_exp = self._predict_minutes(player, stats, fixture) * congestion_mult
            breakdown.minutes_expected += mins_exp

            # Appearance points
            breakdown.appearance_points += self._calc_appearance_points(mins_exp)

            # Goals (including penalties)
            goals_exp, penalty_goals = self._predict_goals_with_penalties(
                player, stats, fixture_pred, mins_exp
            )
            breakdown.goal_expected += goals_exp
            breakdown.goal_points += goals_exp * POINTS.goal_points(player.position_name)

            # Assists
            assists_exp = self._predict_assists(player, stats, fixture_pred, mins_exp)
            breakdown.assist_expected += assists_exp
            breakdown.assist_points += assists_exp * POINTS.ASSIST

            # Clean sheets
            if player.position_name in ('GKP', 'DEF', 'MID'):
                cs_prob = self._predict_clean_sheet(player, fixture_pred, mins_exp)
                breakdown.clean_sheet_prob += cs_prob
                breakdown.clean_sheet_points += cs_prob * POINTS.clean_sheet_points(player.position_name)

            # Saves (GKP only)
            if player.position_name == 'GKP':
                breakdown.save_points += self._predict_save_points(player, stats, fixture_pred, mins_exp)

            # Goals conceded penalty (GKP/DEF)
            if player.position_name in ('GKP', 'DEF'):
                breakdown.goals_conceded_penalty += self._predict_goals_conceded_penalty(fixture_pred, mins_exp)

            # BPS and Bonus points (advanced modeling)
            bps, bonus = self._predict_bonus_advanced(
                player, stats, fixture_pred, mins_exp,
                goals_exp, assists_exp, cs_prob if player.position_name in ('GKP', 'DEF') else 0
            )
            breakdown.expected_bps += bps
            breakdown.bonus_expected += bonus

            # Cards
            breakdown.card_penalty += self._predict_card_penalty(player, stats)

        # Apply form adjustment to attacking returns
        if stats and stats.form_games >= 3:
            form_mult = self._get_form_multiplier(stats)
            breakdown.goal_points *= form_mult
            breakdown.assist_points *= form_mult

        # Total
        breakdown.total_expected_points = (
            breakdown.appearance_points +
            breakdown.goal_points +
            breakdown.assist_points +
            breakdown.clean_sheet_points +
            breakdown.save_points +
            breakdown.bonus_expected +
            breakdown.goals_conceded_penalty +
            breakdown.card_penalty
        )

        # Round to 2 decimals
        breakdown.total_expected_points = round(breakdown.total_expected_points, 2)

        # Confidence based on sample size
        breakdown.confidence = self._calculate_confidence(player, stats)

        # Variance and ceiling/floor for risk assessment
        breakdown.variance = self._calculate_variance(player, stats, breakdown)
        breakdown.ceiling, breakdown.floor = self._calculate_ceiling_floor(breakdown)

        # Differential value (xP adjusted for ownership)
        breakdown.differential_value = self._calculate_differential_value(breakdown)

        return breakdown

    # =========================================================================
    # Availability
    # =========================================================================

    def _is_available(self, player: Player) -> bool:
        """Check if player is available to play."""
        # Status: 'a' = available, 'd' = doubtful, 'i' = injured, 's' = suspended, 'u' = unavailable
        if hasattr(player, 'status'):
            return player.status in ('a', 'd')
        if hasattr(player, 'is_available'):
            return player.is_available
        return True

    # =========================================================================
    # Fixture Prediction
    # =========================================================================

    def _predict_fixture(
        self,
        fixture: Fixture,
        team_id: int,
        teams: Dict[int, Team],
    ) -> FixturePrediction:
        """Predict fixture outcome for xP calculation."""

        cache_key = (fixture.id, team_id, fixture.event)
        if cache_key in self._fixture_cache:
            return self._fixture_cache[cache_key]

        is_home = fixture.team_h == team_id
        opponent_id = fixture.team_a if is_home else fixture.team_h

        # Get team strengths
        team_str = self.team_strength.get(team_id)
        opp_str = self.team_strength.get(opponent_id)

        if team_str and opp_str:
            # Use actual xG data
            if is_home:
                team_xg = team_str.home_xg_per_90 * (opp_str.xga_per_90 / self.league_avg)
                opp_xg = opp_str.away_xg_per_90 * (team_str.xga_per_90 / self.league_avg)
            else:
                team_xg = team_str.away_xg_per_90 * (opp_str.xga_per_90 / self.league_avg)
                opp_xg = opp_str.home_xg_per_90 * (team_str.xga_per_90 / self.league_avg)

            cs_prob = self._poisson_prob(opp_xg, 0)  # P(opponent scores 0)
        else:
            # Fall back to FPL difficulty ratings
            difficulty = fixture.team_h_difficulty if not is_home else fixture.team_a_difficulty

            # Convert difficulty to expected goals
            team_xg = 1.8 - (difficulty - 1) * 0.25  # Range: 0.8 to 1.8
            opp_xg = 0.8 + (difficulty - 1) * 0.25   # Range: 0.8 to 1.8

            if is_home:
                team_xg *= 1.1  # Home advantage
                opp_xg *= 0.9

            cs_prob = self._poisson_prob(opp_xg, 0)

        # Win probability using Poisson
        win_prob = self._calculate_win_prob(team_xg, opp_xg)

        prediction = FixturePrediction(
            fixture_id=fixture.id,
            gameweek=fixture.event,
            team_id=team_id,
            opponent_id=opponent_id,
            is_home=is_home,
            team_xg=team_xg,
            opponent_xg=opp_xg,
            clean_sheet_prob=cs_prob,
            win_prob=win_prob,
            calculated_difficulty=self._xg_to_difficulty(opp_xg),
        )

        self._fixture_cache[cache_key] = prediction
        return prediction

    def _poisson_prob(self, lambda_: float, k: int) -> float:
        """Calculate Poisson probability P(X=k) given lambda."""
        if lambda_ <= 0:
            return 1.0 if k == 0 else 0.0
        return (lambda_ ** k) * math.exp(-lambda_) / math.factorial(k)

    def _calculate_win_prob(self, team_xg: float, opp_xg: float) -> float:
        """Calculate win probability using Poisson model."""
        win_prob = 0.0
        # Sum P(team scores i) * P(opponent scores j) for all i > j
        for i in range(10):  # Team goals
            for j in range(i):  # Opponent goals (less than team)
                win_prob += self._poisson_prob(team_xg, i) * self._poisson_prob(opp_xg, j)
        return win_prob

    def _xg_to_difficulty(self, opp_xg: float) -> float:
        """Convert opponent xG to difficulty rating (1-5 scale)."""
        # opp_xg of ~0.8 = difficulty 1 (easy)
        # opp_xg of ~1.8 = difficulty 5 (hard)
        return max(1, min(5, 1 + (opp_xg - 0.8) * 4))

    # =========================================================================
    # Minutes Prediction with Fixture Congestion
    # =========================================================================

    def _get_congestion_adjustment(
        self,
        player: Player,
        gameweek: int,
        fixtures: List[Fixture],
    ) -> float:
        """
        Reduce minutes/returns if team has heavy fixture load.

        If team plays 3 games in 7 days:
        - Rotation (mins down)
        - Fatigue (performance down ~10%)
        """
        # Count team's fixtures in gameweek-1, gameweek, gameweek+1
        team_fixtures_window = [
            f for f in fixtures
            if (f.team_h == player.team_id or f.team_a == player.team_id)
            and gameweek - 1 <= f.event <= gameweek + 1
        ]

        fixture_count = len(team_fixtures_window)

        if fixture_count >= 3:
            # Heavy congestion
            # Check if player is a nailed starter or rotation risk
            if hasattr(player, 'minutes') and player.minutes > 0:
                games_played = player.minutes / 90
                if games_played >= 10:  # Nailed starter
                    return 0.90  # 10% reduction (fatigue)
                else:
                    return 0.70  # 30% reduction (rotation risk)
            return 0.80  # Default moderate reduction
        elif fixture_count >= 2:
            # Moderate congestion (e.g., DGW)
            return 0.95  # 5% reduction
        else:
            return 1.0  # No adjustment

    def _predict_minutes(
        self,
        player: Player,
        stats: Optional[PlayerStats],
        fixture: Fixture,
    ) -> float:
        """Predict expected minutes for a fixture."""

        # Use advanced stats if available
        if stats and stats.total_minutes > 0:
            avg_mins = stats.avg_minutes_per_game
        elif hasattr(player, 'minutes') and hasattr(player, 'total_points'):
            # Estimate games played from points
            estimated_games = max(1, player.total_points // 2)  # Rough estimate
            if estimated_games > 0:
                avg_mins = min(90, player.minutes / estimated_games)
            else:
                avg_mins = 0.0
        else:
            avg_mins = 60.0  # Default assumption

        # Adjust for availability status
        if hasattr(player, 'status'):
            status_mult = {
                'a': 1.0,   # Available
                'd': 0.5,   # Doubtful (50% chance)
                'i': 0.0,   # Injured
                's': 0.0,   # Suspended
                'u': 0.0,   # Unavailable
            }.get(player.status, 0.5)
            avg_mins *= status_mult

        # Adjust for chance of playing (if available from FPL)
        if hasattr(player, 'chance_of_playing_next_round'):
            cop = player.chance_of_playing_next_round
            if cop is not None:
                avg_mins *= (cop / 100)

        return min(90, avg_mins)

    # =========================================================================
    # Appearance Points
    # =========================================================================

    def _calc_appearance_points(self, minutes_expected: float) -> float:
        """Calculate expected appearance points from minutes."""
        if minutes_expected <= 0:
            return 0.0

        # Probability of playing 60+ mins vs 1-59
        if minutes_expected >= 60:
            prob_60_plus = min(1.0, minutes_expected / 90) * 0.95
            prob_1_to_59 = 0.05
        else:
            prob_60_plus = 0.0
            prob_1_to_59 = min(1.0, minutes_expected / 60)

        return (
            prob_60_plus * POINTS.MINUTES_60_PLUS +
            prob_1_to_59 * POINTS.MINUTES_1_TO_59
        )

    # =========================================================================
    # Goals Prediction (with Penalties)
    # =========================================================================

    def _predict_goals_with_penalties(
        self,
        player: Player,
        stats: Optional[PlayerStats],
        fixture_pred: FixturePrediction,
        minutes_expected: float,
    ) -> Tuple[float, float]:
        """
        Predict expected goals including penalties.

        Returns:
            (total_goals_expected, penalty_goals_expected)
        """
        if minutes_expected <= 0:
            return 0.0, 0.0

        # Open-play xG
        open_play_goals = self._predict_open_play_goals(player, stats, fixture_pred, minutes_expected)

        # Penalty xG (separate calculation)
        penalty_goals = self._predict_penalty_goals(player, stats, fixture_pred, minutes_expected)

        total_goals = open_play_goals + penalty_goals
        return total_goals, penalty_goals

    def _predict_open_play_goals(
        self,
        player: Player,
        stats: Optional[PlayerStats],
        fixture_pred: FixturePrediction,
        minutes_expected: float,
    ) -> float:
        """Predict open-play goals (excluding penalties)."""

        # Get base xG per 90 (excluding penalties if possible)
        if stats and stats.xg_per_90 > 0:
            xg_per_90 = stats.xg_per_90

            # If we have penalty data, subtract penalty xG
            if stats.penalties_taken > 0 and stats.total_minutes > 0:
                # Each penalty is ~0.76 xG
                penalty_xg_per_90 = (stats.penalties_taken / stats.total_minutes) * 90 * 0.76
                xg_per_90 = max(0, xg_per_90 - penalty_xg_per_90)
        else:
            # Fallback: estimate from FPL data
            if hasattr(player, 'expected_goals') and hasattr(player, 'minutes') and player.minutes > 0:
                xg_per_90 = (float(player.expected_goals) / player.minutes) * 90
            else:
                # Position-based defaults
                xg_per_90 = {
                    'GKP': 0.01,
                    'DEF': 0.05,
                    'MID': 0.15,
                    'FWD': 0.35,
                }.get(player.position_name, 0.1)

        # Adjust for fixture difficulty
        difficulty_mult = 2.0 - (fixture_pred.calculated_difficulty / 5)  # 1.6 to 0.6

        # Home advantage
        home_mult = 1.1 if fixture_pred.is_home else 0.95

        # Convert to expected goals for this fixture
        minutes_fraction = minutes_expected / 90
        expected_goals = xg_per_90 * minutes_fraction * difficulty_mult * home_mult

        return expected_goals

    def _predict_penalty_goals(
        self,
        player: Player,
        stats: Optional[PlayerStats],
        fixture_pred: FixturePrediction,
        minutes_expected: float,
    ) -> float:
        """
        Predict expected penalty goals.

        Penalties are different from open-play:
        - Penalties are ~0.76 xG each
        - Need to model: (1) will team get pen? (2) will this player take it?
        """
        if not stats or not stats.is_penalty_taker:
            return 0.0

        # Team's expected penalties this game
        team_str = self.team_strength.get(player.team_id)
        opp_str = self.team_strength.get(fixture_pred.opponent_id)

        if team_str and team_str.penalties_won_per_90 > 0:
            pens_expected = team_str.penalties_won_per_90

            # Adjust for opponent discipline
            if opp_str and opp_str.penalties_conceded_per_90 > 0:
                pens_expected *= (opp_str.penalties_conceded_per_90 / self.league_avg_pens)
        else:
            pens_expected = self.league_avg_pens  # League average

        # Probability this player takes it
        if stats.penalty_order == 1:
            take_prob = 0.95 * (minutes_expected / 90)  # First choice (if on pitch)
        elif stats.penalty_order == 2:
            take_prob = 0.05 * (minutes_expected / 90)  # Only if first choice not on pitch
        else:
            take_prob = 0.0

        # Penalty conversion rate (historical or default)
        if stats.penalties_taken > 0:
            conversion = stats.penalties_scored / stats.penalties_taken
        else:
            conversion = 0.76  # League average

        return pens_expected * take_prob * conversion

    # =========================================================================
    # Assists Prediction
    # =========================================================================

    def _predict_assists(
        self,
        player: Player,
        stats: Optional[PlayerStats],
        fixture_pred: FixturePrediction,
        minutes_expected: float,
    ) -> float:
        """Predict expected assists for the fixture."""

        if minutes_expected <= 0:
            return 0.0

        # Get base xA per 90
        if stats and stats.xa_per_90 > 0:
            xa_per_90 = stats.xa_per_90

            # Boost for set piece takers
            if stats.is_set_piece_taker:
                xa_per_90 *= 1.15  # ~15% boost for corner/FK takers
        else:
            # Fallback: estimate from FPL data
            if hasattr(player, 'expected_assists') and hasattr(player, 'minutes') and player.minutes > 0:
                xa_per_90 = (float(player.expected_assists) / player.minutes) * 90
            else:
                # Position-based defaults
                xa_per_90 = {
                    'GKP': 0.01,
                    'DEF': 0.08,
                    'MID': 0.20,
                    'FWD': 0.15,
                }.get(player.position_name, 0.1)

        # Adjust for team's expected goals in this fixture
        team_xg_mult = fixture_pred.team_xg / 1.4  # Normalize to average

        # Convert to expected assists
        minutes_fraction = minutes_expected / 90
        expected_assists = xa_per_90 * minutes_fraction * team_xg_mult

        return expected_assists

    # =========================================================================
    # Clean Sheet Prediction
    # =========================================================================

    def _predict_clean_sheet(
        self,
        player: Player,
        fixture_pred: FixturePrediction,
        minutes_expected: float,
    ) -> float:
        """Predict clean sheet probability."""

        if minutes_expected < 60:
            # Need 60+ mins for CS points
            return 0.0

        # Base probability from fixture prediction
        cs_prob = fixture_pred.clean_sheet_prob

        # Adjust for minutes (must play 60+)
        prob_60_plus = min(1.0, minutes_expected / 75)

        return cs_prob * prob_60_plus

    # =========================================================================
    # Save Points (GKP)
    # =========================================================================

    def _predict_save_points(
        self,
        player: Player,
        stats: Optional[PlayerStats],
        fixture_pred: FixturePrediction,
        minutes_expected: float,
    ) -> float:
        """Predict save points for goalkeeper."""

        if player.position_name != 'GKP' or minutes_expected < 60:
            return 0.0

        # Expected saves based on opponent shots
        if stats and stats.saves_per_90 > 0:
            saves_per_90 = stats.saves_per_90
        else:
            # Fallback: estimate from opponent xG
            # More opponent xG = more shots = more saves
            saves_per_90 = 2.5 + fixture_pred.opponent_xg  # ~3-4 saves typically

        minutes_fraction = minutes_expected / 90
        expected_saves = saves_per_90 * minutes_fraction

        # Points: 1 per 3 saves
        return expected_saves / POINTS.SAVES_PER_POINT

    # =========================================================================
    # Goals Conceded Penalty (GKP/DEF)
    # =========================================================================

    def _predict_goals_conceded_penalty(
        self,
        fixture_pred: FixturePrediction,
        minutes_expected: float,
    ) -> float:
        """Predict negative points from goals conceded."""

        if minutes_expected < 60:
            return 0.0

        # Expected goals against
        expected_goals_conceded = fixture_pred.opponent_xg * (minutes_expected / 90)

        # Penalty: -1 per 2 goals conceded
        return -expected_goals_conceded / 2

    # =========================================================================
    # Bonus Points (BPS-based)
    # =========================================================================

    def _predict_bonus_advanced(
        self,
        player: Player,
        stats: Optional[PlayerStats],
        fixture_pred: FixturePrediction,
        mins_exp: float,
        goals_exp: float,
        assists_exp: float,
        cs_prob: float,
    ) -> Tuple[float, float]:
        """
        Predict bonus using BPS (Bonus Point System) modeling.

        Returns:
            (expected_bps, expected_bonus_points)
        """
        if mins_exp < 60:
            return 0.0, 0.0

        # Expected BPS from attacking returns
        goal_bps = goals_exp * POINTS.bps_per_goal(player.position_name)
        assist_bps = assists_exp * POINTS.BPS_ASSIST

        # Clean sheet BPS (for GKP/DEF)
        cs_bps = 0.0
        if player.position_name in ('GKP', 'DEF'):
            cs_bps = cs_prob * POINTS.BPS_CLEAN_SHEET_DEF_GKP

        # Expected BPS from underlying actions (if stats available)
        action_bps = 0.0
        if stats:
            minutes_frac = mins_exp / 90

            # Shots on target
            action_bps += stats.shots_on_target_per_90 * POINTS.BPS_SHOT_ON_TARGET * minutes_frac

            # Key passes
            action_bps += stats.key_passes_per_90 * POINTS.BPS_KEY_PASS * minutes_frac

            # Defensive actions (for DEF)
            if player.position_name == 'DEF':
                action_bps += stats.tackles_per_90 * POINTS.BPS_TACKLE_WON * minutes_frac
                action_bps += stats.interceptions_per_90 * POINTS.BPS_INTERCEPTION * minutes_frac
                action_bps += stats.clearances_per_90 * POINTS.BPS_CLEARANCE_BLOCK_INTERCEPTION * minutes_frac

            # Saves (for GKP)
            if player.position_name == 'GKP':
                expected_saves = stats.saves_per_90 * minutes_frac
                action_bps += (expected_saves / 3) * POINTS.BPS_SAVE

        # Goals conceded penalty (for GKP/DEF)
        gc_bps = 0.0
        if player.position_name in ('GKP', 'DEF'):
            expected_gc = fixture_pred.opponent_xg * (mins_exp / 90)
            gc_bps = -(expected_gc / 2) * POINTS.BPS_GOALS_CONCEDED_PER_TWO

        # Total BPS
        total_bps = goal_bps + assist_bps + cs_bps + action_bps + gc_bps

        # Convert BPS to bonus points (simplified model)
        # In reality, need to simulate BPS distribution across both teams
        # Simplified: if BPS > thresholds, likely in bonus
        if total_bps > 40:
            bonus_points = 2.5  # Likely top 3
        elif total_bps > 30:
            bonus_points = 1.5  # Decent chance
        elif total_bps > 25:
            bonus_points = 0.8  # Outside chance
        else:
            bonus_points = 0.2  # Unlikely

        # Use historical average if available and adjust
        if stats and stats.avg_bonus_per_game > 0:
            # Blend model with historical
            historical_bonus = stats.avg_bonus_per_game
            bonus_points = 0.7 * bonus_points + 0.3 * historical_bonus

        return total_bps, bonus_points

    # =========================================================================
    # Card Penalty
    # =========================================================================

    def _predict_card_penalty(
        self,
        player: Player,
        stats: Optional[PlayerStats],
    ) -> float:
        """Predict expected points lost from cards."""

        # Get historical card rate
        if hasattr(player, 'yellow_cards') and hasattr(player, 'red_cards') and hasattr(player, 'minutes'):
            if player.minutes > 0:
                yellow_per_90 = (player.yellow_cards / player.minutes) * 90
                red_per_90 = (player.red_cards / player.minutes) * 90
            else:
                yellow_per_90 = 0.0
                red_per_90 = 0.0
        else:
            # Position-based defaults
            yellow_per_90 = {
                'GKP': 0.02,
                'DEF': 0.15,
                'MID': 0.12,
                'FWD': 0.08,
            }.get(player.position_name, 0.1)
            red_per_90 = yellow_per_90 * 0.02  # ~2% of yellows become reds

        # Expected penalty
        return (
            yellow_per_90 * POINTS.YELLOW_CARD +
            red_per_90 * POINTS.RED_CARD
        )

    # =========================================================================
    # Form Weighting
    # =========================================================================

    def _get_form_multiplier(self, stats: PlayerStats) -> float:
        """
        Weight recent form more heavily than season average.

        70% form (last 5 games), 30% season average.
        """
        if not stats or stats.form_games < 3:
            return 1.0  # Not enough data

        # Compare recent form to season average
        if stats.xgi_per_90 > 0:
            form_ratio = stats.form_xgi_per_90 / stats.xgi_per_90

            # Weight: 70% form, 30% season (so multiplier is between 0.7x and 1.3x of form ratio)
            multiplier = 0.3 + 0.7 * form_ratio

            # Clamp to reasonable range (don't go too extreme)
            return max(0.5, min(1.5, multiplier))

        return 1.0

    # =========================================================================
    # Confidence & Variance
    # =========================================================================

    def _calculate_confidence(
        self,
        player: Player,
        stats: Optional[PlayerStats],
    ) -> float:
        """Calculate confidence in prediction based on sample size."""

        # More games = more confidence
        if hasattr(player, 'minutes') and player.minutes > 0:
            games = player.minutes / 90
        elif stats:
            games = (stats.games_started + stats.games_subbed_on)
        else:
            games = 0

        # Asymptotic approach to 1.0
        # ~0.5 at 5 games, ~0.8 at 15 games, ~0.9 at 30 games
        return 1 - math.exp(-games / 15)

    def _calculate_variance(
        self,
        player: Player,
        stats: Optional[PlayerStats],
        breakdown: ExpectedPointsBreakdown,
    ) -> float:
        """Calculate expected variance in points."""

        if stats and stats.points_variance > 0:
            return stats.points_variance

        # Estimate based on expected returns
        base_variance = 4.0  # Everyone has some baseline variance

        # Goals are high-variance events
        goal_variance = breakdown.goal_expected * 10  # Each expected goal adds variance

        # Assists slightly less so
        assist_variance = breakdown.assist_expected * 6

        return base_variance + goal_variance + assist_variance

    def _calculate_ceiling_floor(
        self,
        breakdown: ExpectedPointsBreakdown,
    ) -> Tuple[float, float]:
        """
        Calculate ceiling (90th percentile) and floor (10th percentile) outcomes.

        Useful for risk-adjusted optimization.
        """
        # Simplified model: assume normal distribution
        # Ceiling = mean + 1.28 * std_dev (90th percentile)
        # Floor = mean - 1.28 * std_dev (10th percentile)

        mean = breakdown.total_expected_points
        std_dev = math.sqrt(breakdown.variance) if breakdown.variance > 0 else 2.0

        ceiling = mean + 1.28 * std_dev
        floor = max(0, mean - 1.28 * std_dev)

        return round(ceiling, 2), round(floor, 2)

    # =========================================================================
    # Differential Value
    # =========================================================================

    def _calculate_differential_value(
        self,
        breakdown: ExpectedPointsBreakdown,
    ) -> float:
        """
        Calculate differential value (xP adjusted for ownership).

        Low-owned players with similar xP have more differential value
        for climbing ranks vs template teams.
        """
        xp = breakdown.total_expected_points
        ownership = breakdown.ownership

        if ownership >= 50:
            # High-owned (template) - no differential value, actually negative
            diff_multiplier = 1.0 - ((ownership - 50) / 200)  # 0.75 to 1.0
        elif ownership >= 20:
            # Medium ownership - neutral
            diff_multiplier = 1.0
        elif ownership >= 5:
            # Low ownership - slight boost
            diff_multiplier = 1.0 + ((20 - ownership) / 100)  # 1.0 to 1.15
        else:
            # Very low ownership - significant differential value
            diff_multiplier = 1.15 + ((5 - ownership) / 50)  # 1.15 to 1.25

        return round(xp * diff_multiplier, 2)


# =============================================================================
# Simple Calculator (Fallback)
# =============================================================================

class SimpleExpectedPointsCalculator(ExpectedPointsCalculator):
    """
    Simple expected points based on form and FPL difficulty.

    Use this when you don't have advanced stats (xG, xA).
    """

    def __init__(
        self,
        elite_predictive_weight: float = 0.90,
        elite_reactive_weight: float = 0.10,
        good_predictive_weight: float = 0.80,
        good_reactive_weight: float = 0.20,
        avg_predictive_weight: float = 0.70,
        avg_reactive_weight: float = 0.30,
    ):
        """
        Initialize calculator with blend ratio parameters.

        Args:
            elite_predictive_weight: Weight for season quality (pts/90) for elite players (>=7 pts/90)
            elite_reactive_weight: Weight for form for elite players
            good_predictive_weight: Weight for season quality for good players (5-7 pts/90)
            good_reactive_weight: Weight for form for good players
            avg_predictive_weight: Weight for season quality for average players (<5 pts/90)
            avg_reactive_weight: Weight for form for average players

        Examples:
            90/10 (very predictive): SimpleExpectedPointsCalculator(0.90, 0.10, 0.80, 0.20, 0.70, 0.30)
            80/20 (balanced): SimpleExpectedPointsCalculator(0.80, 0.20, 0.70, 0.30, 0.60, 0.40)
            70/30 (more reactive): SimpleExpectedPointsCalculator(0.70, 0.30, 0.50, 0.50, 0.40, 0.60)
        """
        self.elite_predictive = elite_predictive_weight
        self.elite_reactive = elite_reactive_weight
        self.good_predictive = good_predictive_weight
        self.good_reactive = good_reactive_weight
        self.avg_predictive = avg_predictive_weight
        self.avg_reactive = avg_reactive_weight

    def calculate(
        self,
        player: Player,
        gameweek: int,
        fixtures: List[Fixture],
        teams: Dict[int, Team],
        captain_mode: bool = False,
    ) -> ExpectedPointsBreakdown:
        """Calculate expected points using basic FPL data.

        Args:
            player: Player to calculate points for
            gameweek: Gameweek to predict
            fixtures: All fixtures
            teams: Team data
            captain_mode: If True, prioritize proven quality over form for captain selection
        """

        # Get player's fixtures
        player_fixtures = [
            f for f in fixtures
            if f.event == gameweek and (f.team_h == player.team_id or f.team_a == player.team_id)
        ]

        if not player_fixtures:
            return ExpectedPointsBreakdown(
                player_id=player.id,
                gameweek=gameweek,
                ownership=player.selected_by_percent,
            )

        # Check availability
        if hasattr(player, 'status') and player.status not in ('a', 'd'):
            return ExpectedPointsBreakdown(
                player_id=player.id,
                gameweek=gameweek,
                ownership=player.selected_by_percent,
            )

        breakdown = ExpectedPointsBreakdown(
            player_id=player.id,
            gameweek=gameweek,
            fixtures=[f.id for f in player_fixtures],
            is_double_gameweek=len(player_fixtures) > 1,
            ownership=player.selected_by_percent,
        )

        # IMPROVED: Blend form with season-long quality
        # This prevents overweighting short-term form over proven performance
        # In captain mode, we prioritize elite player quality even more heavily
        base = self._calculate_baseline(player, captain_mode=captain_mode)

        # Apply position-specific baseline caps BEFORE fixture adjustments
        # Even elite defenders (e.g. 6.4 ppg) shouldn't have baselines >6.5
        # as clean sheets + attacking returns are capped by role
        position_baseline_caps = {
            'GKP': 5.0,   # Keepers max base ~5 (mostly clean sheets + saves)
            'DEF': 6.5,   # Defenders max base ~6.5 (clean sheets + occasional returns)
            'MID': 10.0,  # Midfielders can have higher baselines
            'FWD': 10.0,  # Forwards can have higher baselines
        }
        baseline_cap = position_baseline_caps.get(player.position_name, 8.0)
        base = min(base, baseline_cap)

        total = 0.0

        for fixture in player_fixtures:
            is_home = fixture.team_h == player.team_id
            difficulty = fixture.team_a_difficulty if is_home else fixture.team_h_difficulty

            # Difficulty multiplier
            diff_mult = 1.5 - (difficulty * 0.125)  # 1.375 to 0.875

            # Home/away
            home_mult = 1.08 if is_home else 0.95

            # Position adjustment
            if player.position_name == 'DEF':
                diff_mult = 1.7 - (difficulty * 0.175)  # Defenders more affected
            elif player.position_name == 'FWD':
                diff_mult = 1.3 - (difficulty * 0.075)  # Forwards less affected

            fixture_xp = base * diff_mult * home_mult
            total += fixture_xp

        # Apply chance of playing multiplier (CRITICAL for injury-prone players)
        if hasattr(player, 'chance_of_playing_next_round') and player.chance_of_playing_next_round is not None:
            chance_mult = player.chance_of_playing_next_round / 100
            total *= chance_mult

        # Apply position-specific caps to prevent unrealistic predictions
        # Based on historical FPL maximum realistic single-GW scores
        position_caps = {
            'GKP': 8.0,   # Keepers rarely score >8 in a single GW
            'DEF': 10.0,  # Defenders max ~10 (clean sheet + goal + assists + bonus)
            'MID': 15.0,  # Midfielders can haul bigger
            'FWD': 15.0,  # Forwards can haul bigger
        }
        max_reasonable = position_caps.get(player.position_name, 12.0)
        total = min(total, max_reasonable)

        breakdown.total_expected_points = round(total, 2)
        breakdown.confidence = 0.5  # Lower confidence without advanced stats
        breakdown.variance = 6.0  # Higher variance without detailed modeling
        breakdown.ceiling, breakdown.floor = round(min(total + 6, max_reasonable * 1.5), 2), round(max(0, total - 4), 2)
        breakdown.differential_value = total  # No adjustment without ownership

        return breakdown

    def _calculate_baseline(self, player: Player, captain_mode: bool = False) -> float:
        """
        Calculate baseline expected points using multiple quality indicators.

        Prevents irrational decisions like:
        - Benching Timber (74 pts, 2.7 form) for Wan-Bissaka (17 pts, 3.2 form)
        - Starting Kroupi Jr (33pts, 200 mins) over Isak (19pts, 800 mins)

        Args:
            player: Player to calculate baseline for
            captain_mode: If True, heavily prioritize proven elite quality over form

        Uses: playing time, form, season quality, expected stats, team quality, price, consistency
        """
        # CRITICAL: Filter out players with insufficient playing time
        # Players with < 90 minutes have no reliable baseline
        if player.minutes < 90:
            return 0.0

        # Calculate season-long points per 90 (quality baseline)
        season_quality = 0.0
        if player.minutes >= 90:
            # Points per 90 minutes played
            season_quality = (player.total_points / player.minutes) * 90

        # Get recent form (last 3-5 games)
        form = player.form if player.form > 0 else 0.0

        # Playing time reliability factor (calculate early as it's used below)
        games_played = player.minutes / 90 if player.minutes > 0 else 0

        # Get expected stats if available (xG, xA from FPL API)
        expected_contribution = 0.0
        if hasattr(player, 'expected_goals') and hasattr(player, 'expected_assists'):
            try:
                xg = float(player.expected_goals) if player.expected_goals else 0.0
                xa = float(player.expected_assists) if player.expected_assists else 0.0
                if player.minutes >= 90:
                    # xG/xA per 90
                    xg_per_90 = (xg / player.minutes) * 90
                    xa_per_90 = (xa / player.minutes) * 90

                    # Convert to points
                    goal_points = {'GKP': 6, 'DEF': 6, 'MID': 5, 'FWD': 4}.get(player.position_name, 4)
                    assist_points = xa_per_90 * 3
                    appearance_points = 2.0

                    # Bonus estimate (correlated with goal involvements)
                    bonus_estimate = (xg_per_90 + xa_per_90) * 0.5

                    # Clean sheet points for defenders/keepers
                    cs_points = 0.0
                    if player.position_name in ('GKP', 'DEF'):
                        # Rough estimate: good defenders/keepers get ~40% clean sheets
                        cs_probability = 0.4 if games_played >= 5 else 0.3
                        cs_points = cs_probability * 4

                    expected_contribution = (xg_per_90 * goal_points) + assist_points + appearance_points + bonus_estimate + cs_points
            except (ValueError, TypeError, AttributeError):
                pass

        # CRITICAL: Penalize bench fodder (low minutes) when comparing to regular starters
        # Kroupi Jr might have 33pts in 200 mins = 14.9 pts/90 (amazing!)
        # But Isak has 19pts in 800 mins = 2.1 pts/90 (poor but he's a starter)
        # We should heavily discount Kroupi Jr's inflated per-90 stats
        reliability_factor = 1.0
        if games_played < 3:
            reliability_factor = 0.5  # Very limited minutes = 50% discount
        elif games_played < 5:
            reliability_factor = 0.7  # Some rotation = 30% discount
        elif games_played < 8:
            reliability_factor = 0.85  # Occasional starter = 15% discount
        # else: games_played >= 8 = regular starter, no discount

        # Apply reliability discount to season_quality
        if season_quality > 0:
            season_quality *= reliability_factor

        if games_played < 3:
            # Limited data: heavily weight position baseline, some form
            position_baseline = {
                'GKP': 2.5,
                'DEF': 3.0,
                'MID': 3.5,
                'FWD': 3.5
            }.get(player.position_name, 3.0)

            # Reduce confidence in low-minute players
            return position_baseline * 0.7 if games_played < 2 else position_baseline

        elif games_played < 10:
            # Some data: blend season average with form
            # 50% season, 50% form
            if season_quality > 0:
                return 0.5 * season_quality + 0.5 * form
            else:
                return form if form > 0 else 2.5

        else:
            # Substantial data: PREDICT using season quality + blend with form
            # Season quality (pts/90) already reflects xG/xA converted to actual points
            # The predictive element comes from the 90/10 blend ratio, not from
            # using incomplete expected_contribution

            # Use season quality as baseline (it already incorporates xG/xA indirectly)
            baseline_quality = season_quality if season_quality > 0 else expected_contribution

            if baseline_quality > 0 and form > 0:
                # Adaptive weighting based on player quality
                # Elite players (>7 pts/90): trust underlying stats more (70% xP, 30% form)
                # Good players (5-7 pts/90): balanced (50% xP, 50% form)
                # Average players (<5 pts/90): trust form more (40% xP, 60% form)

                # CAPTAIN MODE: For captain decisions, we care about ceiling not expected value
                # Elite players have higher ceilings, so weight underlying quality even more heavily
                if captain_mode:
                    if baseline_quality >= 7.0:
                        # Elite (Bruno, Salah): 90% underlying, 10% form for captain
                        blended = 0.90 * baseline_quality + 0.10 * form
                    elif baseline_quality >= 5.0:
                        # Good players: 70% underlying, 30% form for captain
                        blended = 0.70 * baseline_quality + 0.30 * form
                    else:
                        # Average players: 50% underlying, 50% form
                        blended = 0.50 * baseline_quality + 0.50 * form
                else:
                    # NORMAL MODE: For team selection and lineup decisions
                    # Use configurable blend ratios (set in __init__)
                    if baseline_quality >= 7.0:
                        # Elite players: use configured weights
                        blended = self.elite_predictive * baseline_quality + self.elite_reactive * form
                    elif baseline_quality >= 5.0:
                        # Good players: use configured weights
                        blended = self.good_predictive * baseline_quality + self.good_reactive * form
                    else:
                        # Average players: use configured weights
                        blended = self.avg_predictive * baseline_quality + self.avg_reactive * form

                # QUALITY FLOOR: Elite players (4+ xP/90) shouldn't drop below 60% of baseline
                # This prevents high-xG players being undervalued during cold streaks
                if baseline_quality >= 4.0:
                    quality_floor = baseline_quality * 0.6
                    blended = max(blended, quality_floor)

                # QUALITY CEILING: Form spikes shouldn't be taken at face value
                # If form >> baseline, it's likely an outlier haul
                # Cap baseline to prevent unrealistic expectations
                if form > baseline_quality * 1.5:
                    # Form is spiking (>1.5x baseline) - cap the upside
                    # Allow some upside but not full spike value
                    blended = min(blended, baseline_quality * 1.3)

                return blended

            elif baseline_quality > 0:
                # No form data, use underlying stats or season average
                return baseline_quality

            elif form > 0:
                # No baseline data (shouldn't happen), use form
                return form

            else:
                # Fallback: points per game
                return player.points_per_game if player.points_per_game > 0 else 2.5


# =============================================================================
# Factory Function
# =============================================================================

def create_calculator(
    player_stats: Optional[Dict[int, PlayerStats]] = None,
    team_strength: Optional[Dict[int, TeamStrength]] = None,
) -> ExpectedPointsCalculator:
    """
    Create appropriate calculator based on available data.

    Args:
        player_stats: Advanced stats from Understat/FBref
        team_strength: Team strength metrics

    Returns:
        AdvancedExpectedPointsCalculator if data available, else Simple
    """
    if player_stats or team_strength:
        logger.info("Using advanced xP calculator with external stats")
        return AdvancedExpectedPointsCalculator(player_stats, team_strength)
    else:
        logger.info("Using simple xP calculator (FPL data only)")
        return SimpleExpectedPointsCalculator()


# =============================================================================
# Compatibility Wrapper (for migration from predictor.py)
# =============================================================================

class ExpectedPointsPredictor:
    """
    Compatibility wrapper around SimpleExpectedPointsCalculator.

    Provides the same API as the old predictor.py for easier migration.
    Use SimpleExpectedPointsCalculator directly for new code.
    """

    def __init__(
        self,
        players: List[Player],
        teams: Dict[int, Team],
        fixtures: List[Fixture]
    ):
        """
        Initialize predictor with game data.

        Args:
            players: List of all players
            teams: Dict mapping team ID to Team
            fixtures: List of fixtures
        """
        self.players = {p.id: p for p in players}
        self.teams = teams
        self.fixtures = fixtures
        self.calculator = SimpleExpectedPointsCalculator()

    def calculate_expected_points(
        self,
        player: Player,
        gameweek: int,
        num_gameweeks: int = 1,
        captain_mode: bool = False
    ) -> float:
        """
        Calculate expected points for a player.

        Args:
            player: Player to calculate for
            gameweek: Starting gameweek
            num_gameweeks: Number of gameweeks (only 1 supported currently)
            captain_mode: If True, prioritize quality over form

        Returns:
            Expected points
        """
        if num_gameweeks != 1:
            logger.warning(f"num_gameweeks={num_gameweeks} not fully supported, using 1")

        breakdown = self.calculator.calculate(
            player=player,
            gameweek=gameweek,
            fixtures=self.fixtures,
            teams=self.teams,
            captain_mode=captain_mode
        )

        return breakdown.total_expected_points