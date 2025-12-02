"""FPL team optimization using linear programming."""

import pulp
from typing import List, Dict, Optional
from .models import Player, SquadConstraints, OptimizedSquad, Transfer, TransferRecommendation


class FPLOptimizer:
    """Optimizes FPL team selection using PuLP linear programming."""

    def __init__(self, players: List[Player], constraints: Optional[SquadConstraints] = None):
        """
        Initialize the optimizer.

        Args:
            players: List of available players to select from.
            constraints: Squad constraints (uses defaults if not provided).
        """
        self.players = players
        self.constraints = constraints or SquadConstraints()
        self.model = None
        self.player_vars = {}

    def optimize_squad(self, objective: str = 'points') -> OptimizedSquad:
        """
        Optimize squad selection.

        Args:
            objective: Optimization objective ('points', 'value', 'form').

        Returns:
            OptimizedSquad with selected players and formation.
        """
        # Create the optimization problem
        self.model = pulp.LpProblem("FPL_Squad_Selection", pulp.LpMaximize)

        # Decision variables: 1 if player is selected, 0 otherwise
        self.player_vars = {
            player.id: pulp.LpVariable(f"player_{player.id}", cat='Binary')
            for player in self.players
        }

        # Starting lineup variables
        starting_vars = {
            player.id: pulp.LpVariable(f"starting_{player.id}", cat='Binary')
            for player in self.players
        }

        # Set objective function based on choice
        objective_values = self._get_objective_values(objective)
        self.model += pulp.lpSum([
            starting_vars[p.id] * objective_values[p.id]
            for p in self.players
        ])

        # Add constraints
        self._add_squad_constraints(self.player_vars, starting_vars)

        # Solve the problem
        self.model.solve(pulp.PULP_CBC_CMD(msg=0))

        # Check if solution was found
        if self.model.status != pulp.LpStatusOptimal:
            raise Exception(f"Optimization failed with status: {pulp.LpStatus[self.model.status]}")

        # Extract solution
        return self._extract_solution(self.player_vars, starting_vars)

    def _get_objective_values(self, objective: str) -> Dict[int, float]:
        """Get objective values for each player based on optimization goal."""
        objective_map = {
            'points': lambda p: p.total_points,
            'value': lambda p: p.total_points / p.cost if p.cost > 0 else 0,
            'form': lambda p: p.form,
            'expected': lambda p: p.expected_goal_involvements,
        }

        objective_func = objective_map.get(objective, objective_map['points'])
        return {player.id: objective_func(player) for player in self.players}

    def _add_squad_constraints(self, player_vars: Dict, starting_vars: Dict):
        """Add all constraints to the optimization model."""
        constraints = self.constraints

        # Squad size constraint
        self.model += (
            pulp.lpSum(player_vars.values()) == constraints.squad_size,
            "squad_size"
        )

        # Starting 11 constraint
        self.model += (
            pulp.lpSum(starting_vars.values()) == constraints.starting_11,
            "starting_11"
        )

        # Budget constraint (cost in tenths, so multiply budget by 10)
        self.model += (
            pulp.lpSum([
                player_vars[p.id] * p.cost for p in self.players
            ]) <= constraints.total_budget * 10,
            "budget"
        )

        # Position constraints for full squad
        position_counts = {
            'GKP': constraints.num_goalkeepers,
            'DEF': constraints.num_defenders,
            'MID': constraints.num_midfielders,
            'FWD': constraints.num_forwards,
        }

        for position, count in position_counts.items():
            self.model += (
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

            self.model += (
                pulp.lpSum([
                    starting_vars[p.id] for p in self.players if p.position_name == position
                ]) >= min_count,
                f"min_starting_{position}"
            )

            self.model += (
                pulp.lpSum([
                    starting_vars[p.id] for p in self.players if p.position_name == position
                ]) <= max_count,
                f"max_starting_{position}"
            )

        # Exactly 1 goalkeeper in starting 11
        self.model += (
            pulp.lpSum([
                starting_vars[p.id] for p in self.players if p.position_name == 'GKP'
            ]) == 1,
            "starting_GKP"
        )

        # Max players per team constraint
        teams = set(p.team_id for p in self.players)
        for team_id in teams:
            self.model += (
                pulp.lpSum([
                    player_vars[p.id] for p in self.players if p.team_id == team_id
                ]) <= constraints.max_players_per_team,
                f"max_from_team_{team_id}"
            )

        # Link squad and starting lineup: can't start if not in squad
        for player in self.players:
            self.model += (
                starting_vars[player.id] <= player_vars[player.id],
                f"link_squad_starting_{player.id}"
            )

    def _extract_solution(self, player_vars: Dict, starting_vars: Dict) -> OptimizedSquad:
        """Extract the optimized squad from the solution."""
        # Get selected players
        selected_players = [
            p for p in self.players
            if pulp.value(player_vars[p.id]) == 1
        ]

        # Get starting 11 player IDs
        starting_ids = [
            p.id for p in self.players
            if pulp.value(starting_vars[p.id]) == 1
        ]

        # Calculate formation
        starting_players = [p for p in selected_players if p.id in starting_ids]
        formation = self._calculate_formation(starting_players)

        # Calculate total cost and expected points
        total_cost = sum(p.cost for p in selected_players)
        expected_points = sum(p.total_points for p in starting_players)

        # Select captain (highest points in starting 11)
        captain = max(starting_players, key=lambda p: p.total_points)
        # Vice captain (second highest)
        vice_captain = sorted(starting_players, key=lambda p: p.total_points, reverse=True)[1]

        return OptimizedSquad(
            players=selected_players,
            starting_11_ids=starting_ids,
            captain_id=captain.id,
            vice_captain_id=vice_captain.id,
            total_cost=total_cost,
            expected_points=expected_points,
            formation=formation,
        )

    def _calculate_formation(self, starting_players: List[Player]) -> str:
        """Calculate formation string (e.g., '3-4-3') from starting players."""
        def_count = sum(1 for p in starting_players if p.position_name == 'DEF')
        mid_count = sum(1 for p in starting_players if p.position_name == 'MID')
        fwd_count = sum(1 for p in starting_players if p.position_name == 'FWD')

        return f"{def_count}-{mid_count}-{fwd_count}"

    def optimize_with_existing_players(
        self,
        existing_player_ids: List[int],
        num_transfers: int = 1
    ) -> OptimizedSquad:
        """
        Optimize squad with existing players (for transfers).

        Args:
            existing_player_ids: IDs of players already in squad.
            num_transfers: Number of transfers allowed.

        Returns:
            OptimizedSquad with minimal transfers.
        """
        # Create the optimization problem
        self.model = pulp.LpProblem("FPL_Transfer_Optimization", pulp.LpMaximize)

        # Decision variables
        self.player_vars = {
            player.id: pulp.LpVariable(f"player_{player.id}", cat='Binary')
            for player in self.players
        }

        starting_vars = {
            player.id: pulp.LpVariable(f"starting_{player.id}", cat='Binary')
            for player in self.players
        }

        # Objective: maximize points
        self.model += pulp.lpSum([
            starting_vars[p.id] * p.total_points
            for p in self.players
        ])

        # Add standard constraints
        self._add_squad_constraints(self.player_vars, starting_vars)

        # Transfer constraint: limit changes from existing squad
        transfers_out = pulp.lpSum([
            1 - self.player_vars[pid]
            for pid in existing_player_ids
            if any(p.id == pid for p in self.players)
        ])

        self.model += (
            transfers_out <= num_transfers,
            "max_transfers"
        )

        # Solve
        self.model.solve(pulp.PULP_CBC_CMD(msg=0))

        return self._extract_solution(self.player_vars, starting_vars)