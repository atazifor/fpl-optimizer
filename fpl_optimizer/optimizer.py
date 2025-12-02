"""Advanced FPL team optimization with multi-gameweek planning."""

import pulp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Set, Tuple
from .models import Player, SquadConstraints, OptimizedSquad, Fixture, Team
from .expected_points import (
    ExpectedPointsCalculator,
    AdvancedExpectedPointsCalculator,
    SimpleExpectedPointsCalculator,
    create_calculator,
)


class ObjectiveType(Enum):
    """Optimization objective types."""
    POINTS = "points"  # Maximize expected points
    VALUE = "value"  # Maximize points per £
    DIFFERENTIAL = "differential"  # Target low ownership players
    SAFETY = "safety"  # Target high ownership (template)


class Chip(Enum):
    """Available FPL chips."""
    WILDCARD = "wildcard"
    BENCH_BOOST = "bench_boost"
    TRIPLE_CAPTAIN = "triple_captain"
    FREE_HIT = "free_hit"


@dataclass
class OptimizationConfig:
    """Configuration for multi-gameweek optimization."""
    horizon_weeks: int = 2  # How many weeks to plan ahead
    transfer_penalty: int = 4  # Points cost per transfer
    decay_rate: float = 0.9  # Future weeks are worth less
    min_expected_minutes: int = 60  # Filter out rotation risks
    max_ownership_differential: float = 10.0  # Max ownership % for differentials
    free_transfers: int = 1  # Available free transfers this week
    bank: float = 0.0  # Money in the bank
    objective: ObjectiveType = ObjectiveType.POINTS
    max_transfers: Optional[int] = None  # Force specific number of transfers (None = auto)


@dataclass
class GameweekPrediction:
    """Expected points for a player in a specific gameweek."""
    player_id: int
    gameweek: int
    expected_points: float
    fixtures: List[int]  # Fixture IDs


@dataclass
class TransferPlan:
    """Multi-gameweek transfer plan."""
    gameweek: int
    transfers_in: List[int]  # Player IDs to bring in
    transfers_out: List[int]  # Player IDs to remove
    transfer_cost: int  # Points penalty
    expected_gain: float
    chip_to_use: Optional[Chip] = None


@dataclass
class OptimizationResult:
    """Result of multi-gameweek optimization."""
    current_squad: OptimizedSquad
    transfer_plans: List[TransferPlan]
    total_expected_points: float
    horizon_breakdown: Dict[int, float]  # Expected points per GW
    chip_recommendations: Dict[Chip, int]  # Chip -> recommended GW


# Note: ExpectedPointsCalculator classes are now imported from expected_points module


class FPLOptimizer:
    """Advanced FPL optimizer with multi-gameweek planning."""

    def __init__(
        self,
        players: List[Player],
        fixtures: List[Fixture],
        teams: Dict[int, Team],
        constraints: Optional[SquadConstraints] = None,
        xp_calculator: Optional[ExpectedPointsCalculator] = None
    ):
        """
        Initialize the advanced optimizer.

        Args:
            players: List of available players.
            fixtures: List of fixtures for prediction.
            teams: Dict of team_id -> Team for strength calculations.
            constraints: Squad constraints (uses defaults if not provided).
            xp_calculator: Expected points calculator (uses simple if not provided).
        """
        self.players = players
        self.fixtures = fixtures
        self.teams = teams
        self.constraints = constraints or SquadConstraints()
        self.xp_calculator = xp_calculator or SimpleExpectedPointsCalculator()
        self.model = None
        self.player_vars = {}
        self._xp_cache: Dict[Tuple[int, int], float] = {}

    def optimize_squad(
        self,
        config: OptimizationConfig = None
    ) -> OptimizedSquad:
        """
        Optimize a fresh squad (no existing players).

        Args:
            config: Optimization configuration.

        Returns:
            OptimizedSquad with best team.
        """
        config = config or OptimizationConfig()

        # Precompute expected points for next gameweek
        current_gw = self._get_current_gameweek()
        self._precompute_expected_points(current_gw, current_gw)

        # Create the optimization problem
        self.model = pulp.LpProblem("FPL_Fresh_Squad", pulp.LpMaximize)

        # Decision variables
        self.player_vars = {
            player.id: pulp.LpVariable(f"player_{player.id}", cat='Binary')
            for player in self.players
        }

        starting_vars = {
            player.id: pulp.LpVariable(f"starting_{player.id}", cat='Binary')
            for player in self.players
        }

        captain_vars = {
            player.id: pulp.LpVariable(f"captain_{player.id}", cat='Binary')
            for player in self.players
        }

        # Objective function based on expected points
        objective_values = self._get_objective_values(config, current_gw)

        self.model += pulp.lpSum([
            starting_vars[p.id] * objective_values[p.id] +
            captain_vars[p.id] * objective_values[p.id]  # Captain gets 2x points
            for p in self.players
        ])

        # Add standard constraints
        self._add_squad_constraints(self.player_vars, starting_vars)
        self._add_captain_constraints(captain_vars, starting_vars)

        # Solve
        self.model.solve(pulp.PULP_CBC_CMD(msg=0))

        if self.model.status != pulp.LpStatusOptimal:
            raise Exception(f"Optimization failed: {pulp.LpStatus[self.model.status]}")

        return self._extract_solution(self.player_vars, starting_vars, captain_vars, current_gw)

    def optimize_with_transfers(
        self,
        existing_player_ids: List[int],
        config: OptimizationConfig = None
    ) -> OptimizationResult:
        """
        Optimize transfers with multi-gameweek planning.

        Args:
            existing_player_ids: Current squad player IDs.
            config: Optimization configuration.

        Returns:
            OptimizationResult with transfer plan and projections.
        """
        config = config or OptimizationConfig()

        # Validation
        if len(existing_player_ids) != self.constraints.squad_size:
            raise ValueError(
                f"Expected {self.constraints.squad_size} players, got {len(existing_player_ids)}"
            )

        unknown_ids = set(existing_player_ids) - {p.id for p in self.players}
        if unknown_ids:
            raise ValueError(f"Unknown player IDs: {unknown_ids}")

        current_gw = self._get_current_gameweek()

        # Precompute expected points for horizon
        self._precompute_expected_points(
            current_gw,
            current_gw + config.horizon_weeks - 1
        )

        # Build transfer plan
        transfer_plans = []
        current_squad = existing_player_ids.copy()
        first_week_squad_ids = existing_player_ids.copy()  # Track first week's result separately
        total_expected = 0.0
        horizon_breakdown = {}

        for week_offset in range(config.horizon_weeks):
            gw = current_gw + week_offset

            # Determine number of transfers for this week
            if week_offset == 0:
                if config.max_transfers is not None:
                    # Use forced number of transfers
                    max_transfers = config.max_transfers
                else:
                    # Auto-decide: can take a hit
                    max_transfers = min(config.free_transfers + 1, 15)
            else:
                max_transfers = 1  # Plan for 1 FT in future weeks

            # Optimize for this gameweek
            squad, transfers_in, transfers_out = self._optimize_single_week(
                current_squad=current_squad,
                gameweek=gw,
                max_transfers=max_transfers,
                config=config
            )

            # Calculate transfer cost
            num_transfers = len(transfers_in)
            free_transfers = config.free_transfers if week_offset == 0 else 1
            hits = max(0, num_transfers - free_transfers)
            transfer_cost = hits * config.transfer_penalty

            # Calculate expected points for this week
            week_expected = sum(
                self._get_xp(p_id, gw) for p_id in squad
            )

            # Apply decay for future weeks
            decay = config.decay_rate ** week_offset
            week_expected *= decay

            total_expected += week_expected - transfer_cost
            horizon_breakdown[gw] = week_expected

            # Create transfer plan
            if transfers_in:
                transfer_plans.append(TransferPlan(
                    gameweek=gw,
                    transfers_in=transfers_in,
                    transfers_out=transfers_out,
                    transfer_cost=transfer_cost,
                    expected_gain=week_expected
                ))

            # Save first week's squad for return value
            if week_offset == 0:
                first_week_squad_ids = squad

            # Update current squad for next iteration
            current_squad = squad

        # Get the NEW squad after first week's transfers (not the original squad)
        new_squad_obj = self._build_squad_from_ids(
            first_week_squad_ids,
            current_gw,
            config
        )

        return OptimizationResult(
            current_squad=new_squad_obj,
            transfer_plans=transfer_plans,
            total_expected_points=total_expected,
            horizon_breakdown=horizon_breakdown,
            chip_recommendations=self._recommend_chips(current_gw, config.horizon_weeks)
        )

    def _optimize_single_week(
        self,
        current_squad: List[int],
        gameweek: int,
        max_transfers: int,
        config: OptimizationConfig
    ) -> Tuple[List[int], List[int], List[int]]:
        """Optimize squad for a single gameweek with transfer limit."""

        # Create optimization problem
        model = pulp.LpProblem(f"FPL_GW{gameweek}", pulp.LpMaximize)

        # Decision variables
        player_vars = {
            p.id: pulp.LpVariable(f"p_{p.id}", cat='Binary')
            for p in self.players
        }

        starting_vars = {
            p.id: pulp.LpVariable(f"s_{p.id}", cat='Binary')
            for p in self.players
        }

        captain_vars = {
            p.id: pulp.LpVariable(f"c_{p.id}", cat='Binary')
            for p in self.players
        }

        # Objective: maximize expected points
        objective_values = self._get_objective_values(config, gameweek)

        model += pulp.lpSum([
            starting_vars[p.id] * objective_values[p.id] +
            captain_vars[p.id] * objective_values[p.id]
            for p in self.players
        ])

        # Standard constraints
        self._add_squad_constraints_to_model(model, player_vars, starting_vars)
        self._add_captain_constraints_to_model(model, captain_vars, starting_vars)

        # Transfer constraint
        transfers_out = pulp.lpSum([
            1 - player_vars[pid]
            for pid in current_squad
            if any(p.id == pid for p in self.players)
        ])

        model += (transfers_out <= max_transfers, "max_transfers")

        # Solve
        model.solve(pulp.PULP_CBC_CMD(msg=0))

        if model.status != pulp.LpStatusOptimal:
            # If can't solve, return current squad
            return current_squad, [], []

        # Extract solution
        new_squad = [p.id for p in self.players if pulp.value(player_vars[p.id]) == 1]

        # Determine transfers
        transfers_out = [pid for pid in current_squad if pid not in new_squad]
        transfers_in = [pid for pid in new_squad if pid not in current_squad]

        return new_squad, transfers_in, transfers_out

    def _add_squad_constraints(self, player_vars: Dict, starting_vars: Dict):
        """Add all standard squad constraints to self.model."""
        self._add_squad_constraints_to_model(self.model, player_vars, starting_vars)

    def _add_squad_constraints_to_model(self, model, player_vars: Dict, starting_vars: Dict):
        """Add all standard squad constraints to given model."""
        constraints = self.constraints

        # Squad size
        model += (
            pulp.lpSum(player_vars.values()) == constraints.squad_size,
            "squad_size"
        )

        # Starting 11
        model += (
            pulp.lpSum(starting_vars.values()) == constraints.starting_11,
            "starting_11"
        )

        # Budget (cost in tenths)
        model += (
            pulp.lpSum([
                player_vars[p.id] * p.cost for p in self.players
            ]) <= constraints.total_budget * 10,
            "budget"
        )

        # Position constraints
        position_counts = {
            'GKP': constraints.num_goalkeepers,
            'DEF': constraints.num_defenders,
            'MID': constraints.num_midfielders,
            'FWD': constraints.num_forwards,
        }

        for position, count in position_counts.items():
            model += (
                pulp.lpSum([
                    player_vars[p.id] for p in self.players if p.position_name == position
                ]) == count,
                f"squad_{position}"
            )

        # Starting lineup position constraints
        for position in ['DEF', 'MID', 'FWD']:
            min_attr = f"min_starting_{position.lower()}s"
            max_attr = f"max_starting_{position.lower()}s"
            min_count = getattr(constraints, min_attr, 0)
            max_count = getattr(constraints, max_attr, 11)

            model += (
                pulp.lpSum([
                    starting_vars[p.id] for p in self.players if p.position_name == position
                ]) >= min_count,
                f"min_starting_{position}"
            )

            model += (
                pulp.lpSum([
                    starting_vars[p.id] for p in self.players if p.position_name == position
                ]) <= max_count,
                f"max_starting_{position}"
            )

        # Exactly 1 GKP in starting 11
        model += (
            pulp.lpSum([
                starting_vars[p.id] for p in self.players if p.position_name == 'GKP'
            ]) == 1,
            "starting_GKP"
        )

        # Max players per team
        teams = set(p.team_id for p in self.players)
        for team_id in teams:
            model += (
                pulp.lpSum([
                    player_vars[p.id] for p in self.players if p.team_id == team_id
                ]) <= constraints.max_players_per_team,
                f"max_from_team_{team_id}"
            )

        # Link squad and starting
        for player in self.players:
            model += (
                starting_vars[player.id] <= player_vars[player.id],
                f"link_{player.id}"
            )

    def _add_captain_constraints(self, captain_vars: Dict, starting_vars: Dict):
        """Add captain constraints to self.model."""
        self._add_captain_constraints_to_model(self.model, captain_vars, starting_vars)

    def _add_captain_constraints_to_model(self, model, captain_vars: Dict, starting_vars: Dict):
        """Add captain constraints to given model."""
        # Exactly 1 captain
        model += (
            pulp.lpSum(captain_vars.values()) == 1,
            "one_captain"
        )

        # Captain must be in starting 11
        for player in self.players:
            model += (
                captain_vars[player.id] <= starting_vars[player.id],
                f"captain_must_start_{player.id}"
            )

    def _get_objective_values(self, config: OptimizationConfig, gameweek: int) -> Dict[int, float]:
        """Get objective values for each player based on config."""
        values = {}

        for player in self.players:
            xp = self._get_xp(player.id, gameweek)

            if config.objective == ObjectiveType.POINTS:
                values[player.id] = xp
            elif config.objective == ObjectiveType.VALUE:
                values[player.id] = (xp / player.cost) * 10 if player.cost > 0 else 0
            elif config.objective == ObjectiveType.DIFFERENTIAL:
                # Boost low ownership players
                ownership_penalty = player.selected_by_percent / 100
                values[player.id] = xp * (1 - ownership_penalty * 0.5)
            elif config.objective == ObjectiveType.SAFETY:
                # Boost high ownership players
                ownership_boost = player.selected_by_percent / 100
                values[player.id] = xp * (1 + ownership_boost * 0.3)

        return values

    def _precompute_expected_points(self, start_gw: int, end_gw: int):
        """Precompute expected points for all players across gameweeks."""
        for gw in range(start_gw, end_gw + 1):
            for player in self.players:
                if (player.id, gw) not in self._xp_cache:
                    result = self.xp_calculator.calculate(player, gw, self.fixtures, self.teams)
                    # Handle both breakdown objects and floats for backwards compatibility
                    if hasattr(result, 'total_expected_points'):
                        xp = result.total_expected_points
                    else:
                        xp = float(result)
                    self._xp_cache[(player.id, gw)] = xp

    def _get_xp(self, player_id: int, gameweek: int) -> float:
        """Get cached expected points for a player in a gameweek."""
        return self._xp_cache.get((player_id, gameweek), 0.0)

    def _extract_solution(
        self,
        player_vars: Dict,
        starting_vars: Dict,
        captain_vars: Dict,
        gameweek: int
    ) -> OptimizedSquad:
        """Extract optimized squad from solution."""

        # Get selected players
        selected_players = [
            p for p in self.players
            if pulp.value(player_vars[p.id]) == 1
        ]

        # Get starting 11 IDs
        starting_ids = [
            p.id for p in self.players
            if pulp.value(starting_vars[p.id]) == 1
        ]

        # Get captain
        captain_id = next(
            (p.id for p in self.players if pulp.value(captain_vars[p.id]) == 1),
            starting_ids[0] if starting_ids else 0
        )

        # Vice captain (second highest xP in starting 11)
        starting_players = [p for p in selected_players if p.id in starting_ids]
        starting_xp = [(p.id, self._get_xp(p.id, gameweek)) for p in starting_players]
        starting_xp.sort(key=lambda x: x[1], reverse=True)
        vice_captain_id = starting_xp[1][0] if len(starting_xp) > 1 else captain_id

        # Calculate formation
        formation = self._calculate_formation(starting_players)

        # Calculate total cost and expected points
        total_cost = sum(p.cost for p in selected_players)
        expected_points = sum(self._get_xp(p.id, gameweek) for p in starting_players)
        expected_points += self._get_xp(captain_id, gameweek)  # Captain bonus

        return OptimizedSquad(
            players=selected_players,
            starting_11_ids=starting_ids,
            captain_id=captain_id,
            vice_captain_id=vice_captain_id,
            total_cost=total_cost,
            expected_points=expected_points,
            formation=formation,
        )

    def _build_squad_from_ids(
        self,
        player_ids: List[int],
        gameweek: int,
        config: OptimizationConfig
    ) -> OptimizedSquad:
        """Build OptimizedSquad from player IDs with valid formation."""

        players = [p for p in self.players if p.id in player_ids]

        # Group players by position with their xP
        by_position = {
            'GKP': [],
            'DEF': [],
            'MID': [],
            'FWD': []
        }

        for p in players:
            xp = self._get_xp(p.id, gameweek)
            by_position[p.position_name].append((p, xp))

        # Sort each position by xP
        for pos in by_position:
            by_position[pos].sort(key=lambda x: x[1], reverse=True)

        # Build starting 11 respecting formation constraints
        starting = []

        # Always 1 GKP
        if by_position['GKP']:
            starting.append(by_position['GKP'][0][0])

        # Get constraints
        min_def = self.constraints.min_starting_defenders
        min_mid = self.constraints.min_starting_midfielders
        min_fwd = self.constraints.min_starting_forwards
        max_def = self.constraints.max_starting_defenders
        max_mid = self.constraints.max_starting_midfielders
        max_fwd = self.constraints.max_starting_forwards

        # Start with minimum required from each position
        for p, _ in by_position['DEF'][:min_def]:
            starting.append(p)
        for p, _ in by_position['MID'][:min_mid]:
            starting.append(p)
        for p, _ in by_position['FWD'][:min_fwd]:
            starting.append(p)

        # Fill remaining spots with best available players
        remaining_slots = 11 - len(starting)

        # Create pool of remaining players
        remaining_pool = []
        remaining_pool.extend(by_position['DEF'][min_def:max_def])
        remaining_pool.extend(by_position['MID'][min_mid:max_mid])
        remaining_pool.extend(by_position['FWD'][min_fwd:max_fwd])

        # Sort by xP and take best
        remaining_pool.sort(key=lambda x: x[1], reverse=True)
        for p, _ in remaining_pool[:remaining_slots]:
            starting.append(p)

        starting_ids = [p.id for p in starting]

        # Captain and vice captain (highest xP in starting 11)
        starting_xp = [(p.id, self._get_xp(p.id, gameweek)) for p in starting if p.position_name != 'GKP']
        starting_xp.sort(key=lambda x: x[1], reverse=True)

        captain_id = starting_xp[0][0] if starting_xp else starting_ids[0]
        vice_captain_id = starting_xp[1][0] if len(starting_xp) > 1 else captain_id

        formation = self._calculate_formation(starting)
        total_cost = sum(p.cost for p in players)
        expected_points = sum(self._get_xp(p.id, gameweek) for p in starting)
        expected_points += self._get_xp(captain_id, gameweek)

        return OptimizedSquad(
            players=players,
            starting_11_ids=starting_ids,
            captain_id=captain_id,
            vice_captain_id=vice_captain_id,
            total_cost=total_cost,
            expected_points=expected_points,
            formation=formation,
        )

    def _calculate_formation(self, starting_players: List[Player]) -> str:
        """Calculate formation string from starting players."""
        def_count = sum(1 for p in starting_players if p.position_name == 'DEF')
        mid_count = sum(1 for p in starting_players if p.position_name == 'MID')
        fwd_count = sum(1 for p in starting_players if p.position_name == 'FWD')
        return f"{def_count}-{mid_count}-{fwd_count}"

    def _get_current_gameweek(self) -> int:
        """Get current gameweek number from fixtures."""
        if not self.fixtures:
            return 1

        # Find the next unfinished gameweek
        upcoming = [f.event for f in self.fixtures if not f.finished]
        return min(upcoming) if upcoming else max(f.event for f in self.fixtures)

    def _recommend_chips(self, current_gw: int, horizon: int) -> Dict[Chip, int]:
        """Recommend when to use chips (placeholder for now)."""
        # This would analyze fixtures for DGW/BGW
        # For now, return empty recommendations
        return {}