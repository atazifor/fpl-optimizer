"""Expected points calculation using form and fixture difficulty."""

from typing import List, Dict
from .models import Player, Team, Fixture, PlayerHistory


class ExpectedPointsPredictor:
    """Predicts expected points for players based on form and fixtures."""

    def __init__(
        self,
        players: List[Player],
        teams: Dict[int, Team],
        fixtures: List[Fixture]
    ):
        """
        Initialize the predictor.

        Args:
            players: List of all players
            teams: Dictionary mapping team ID to Team
            fixtures: List of upcoming fixtures
        """
        self.players = {p.id: p for p in players}
        self.teams = teams
        self.fixtures = fixtures

        # Weights for form calculation (more recent games weighted higher)
        self.form_weights = [0.35, 0.25, 0.20, 0.12, 0.08]  # Last 5 games

    def calculate_expected_points(
        self,
        player: Player,
        gameweek: int,
        num_gameweeks: int = 1
    ) -> float:
        """
        Calculate expected points for a player over next N gameweeks.

        Args:
            player: Player to calculate for
            gameweek: Starting gameweek
            num_gameweeks: Number of gameweeks to predict

        Returns:
            Expected points
        """
        if not player.is_available:
            return 0.0

        # Get player's upcoming fixtures
        player_fixtures = self._get_player_fixtures(player.team_id, gameweek, num_gameweeks)

        if not player_fixtures:
            return 0.0

        # Base expected points from form
        form_points = self._calculate_form_points(player)

        # Adjust for fixture difficulty
        fixture_adjusted_points = 0.0

        for fixture in player_fixtures:
            difficulty_multiplier = self._get_fixture_difficulty_multiplier(
                player, fixture
            )
            fixture_adjusted_points += form_points * difficulty_multiplier

        # Average over multiple gameweeks
        expected = fixture_adjusted_points / len(player_fixtures) if player_fixtures else 0.0

        # Apply minutes prediction (reduce if rotation risk)
        minutes_multiplier = self._get_minutes_multiplier(player)
        expected *= minutes_multiplier

        return round(expected, 2)

    def _calculate_form_points(self, player: Player) -> float:
        """
        Calculate base expected points from player's form.

        Uses weighted average of recent points, with more weight on recent games.

        Args:
            player: Player to calculate form for

        Returns:
            Expected points per game based on form
        """
        # Use the form field which is already a weighted average
        if player.form > 0:
            return float(player.form)

        # Fallback to points per game if form not available
        if player.points_per_game > 0:
            return player.points_per_game

        # Last resort: use total points averaged over season
        # Assume ~15 games played at midseason
        if player.total_points > 0:
            estimated_games = max(player.minutes // 90, 1)
            return player.total_points / estimated_games

        return 0.0

    def _get_player_fixtures(
        self,
        team_id: int,
        start_gameweek: int,
        num_gameweeks: int
    ) -> List[Fixture]:
        """Get upcoming fixtures for a team."""
        end_gameweek = start_gameweek + num_gameweeks

        return [
            f for f in self.fixtures
            if (f.team_h == team_id or f.team_a == team_id)
            and start_gameweek <= f.event < end_gameweek
            and not f.finished
        ]

    def _get_fixture_difficulty_multiplier(
        self,
        player: Player,
        fixture: Fixture
    ) -> float:
        """
        Calculate fixture difficulty multiplier.

        Easier fixtures = higher multiplier (more expected points)
        Harder fixtures = lower multiplier (fewer expected points)

        Args:
            player: Player in question
            fixture: Fixture to analyze

        Returns:
            Multiplier for expected points (0.6 to 1.4)
        """
        is_home = fixture.team_h == player.team_id

        # Get opponent's defensive strength
        opponent_id = fixture.team_a if is_home else fixture.team_h
        opponent = self.teams.get(opponent_id)

        if not opponent:
            return 1.0  # Neutral if opponent data missing

        # For attacking players, look at opponent's defensive strength
        if player.position in [3, 4]:  # MID or FWD
            opponent_strength = (
                opponent.strength_defence_home if not is_home
                else opponent.strength_defence_away
            )
            # Weaker defense = easier to score against = higher multiplier
            # Scale: 1 (weakest) to 5 (strongest) -> 1.4 to 0.6 multiplier
            multiplier = 1.0 + (3 - opponent_strength) * 0.2

        # For defensive players (GKP, DEF), look at opponent's attacking strength
        elif player.position in [1, 2]:  # GKP or DEF
            opponent_strength = (
                opponent.strength_attack_home if not is_home
                else opponent.strength_attack_away
            )
            # Weaker attack = easier to keep clean sheet = higher multiplier
            multiplier = 1.0 + (3 - opponent_strength) * 0.2

        else:
            multiplier = 1.0

        # Also consider FPL's own difficulty rating
        fpl_difficulty = fixture.team_h_difficulty if is_home else fixture.team_a_difficulty

        # FPL difficulty: 1 (easiest) to 5 (hardest)
        # Adjust multiplier based on FPL difficulty
        fpl_adjustment = (3 - fpl_difficulty) * 0.1
        multiplier += fpl_adjustment

        # Clamp between 0.6 and 1.4
        return max(0.6, min(1.4, multiplier))

    def _get_minutes_multiplier(self, player: Player) -> float:
        """
        Calculate multiplier based on expected minutes played.

        Players with rotation risk or injury concerns get reduced multiplier.

        Args:
            player: Player to analyze

        Returns:
            Multiplier between 0.0 and 1.0
        """
        # Check injury/availability
        if player.chance_of_playing_next_round is not None:
            chance = player.chance_of_playing_next_round / 100.0
            return chance

        # Check if player is a regular starter (played >75% of available minutes)
        # Assume 90 minutes per game, estimate total possible minutes
        if player.minutes == 0:
            return 0.5  # Unknown, assume moderate risk

        # Rough estimate: if playing time suggests regular starter
        # High minutes = regular starter = 1.0
        # Low minutes = rotation risk = 0.5-0.8
        games_played = player.minutes / 90

        if games_played >= 10:  # Regular starter
            return 1.0
        elif games_played >= 5:  # Squad player
            return 0.85
        elif games_played >= 2:  # Rotation player
            return 0.6
        else:  # Rarely plays
            return 0.3

    def calculate_captain_value(self, player: Player, gameweek: int) -> float:
        """
        Calculate expected points for a player as captain (points x2).

        Args:
            player: Player to consider for captaincy
            gameweek: Gameweek number

        Returns:
            Expected captain points (2x regular expected points)
        """
        base_expected = self.calculate_expected_points(player, gameweek, num_gameweeks=1)
        return base_expected * 2

    def get_best_captain(
        self,
        squad_players: List[Player],
        gameweek: int
    ) -> Player:
        """
        Identify best captain choice from squad.

        Args:
            squad_players: Players in the squad
            gameweek: Gameweek number

        Returns:
            Best player to captain
        """
        captain_scores = [
            (player, self.calculate_captain_value(player, gameweek))
            for player in squad_players
        ]

        # Sort by expected captain points
        captain_scores.sort(key=lambda x: x[1], reverse=True)

        return captain_scores[0][0] if captain_scores else squad_players[0]

    def calculate_fixture_difficulty_rating(
        self,
        team_id: int,
        gameweek: int,
        num_gameweeks: int = 5
    ) -> float:
        """
        Calculate fixture difficulty rating for a team over next N gameweeks.

        Lower rating = easier fixtures.

        Args:
            team_id: Team ID
            gameweek: Starting gameweek
            num_gameweeks: Number of gameweeks to analyze

        Returns:
            Average fixture difficulty (1-5 scale)
        """
        fixtures = self._get_player_fixtures(team_id, gameweek, num_gameweeks)

        if not fixtures:
            return 3.0  # Neutral

        difficulties = []
        for fixture in fixtures:
            if fixture.team_h == team_id:
                difficulties.append(fixture.team_h_difficulty)
            else:
                difficulties.append(fixture.team_a_difficulty)

        return sum(difficulties) / len(difficulties) if difficulties else 3.0