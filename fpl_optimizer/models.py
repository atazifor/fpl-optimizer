"""Pydantic data models for FPL entities."""

from typing import List, Optional
from pydantic import BaseModel, Field, computed_field


class Player(BaseModel):
    """Represents an FPL player with stats and pricing."""

    id: int
    name: str = Field(alias='web_name')
    full_name: str = Field(alias='first_name')
    last_name: str = Field(alias='second_name')
    team_id: int = Field(alias='team')
    position: int = Field(alias='element_type')  # 1=GKP, 2=DEF, 3=MID, 4=FWD
    cost: float = Field(alias='now_cost')  # Cost in tenths (e.g., 95 = £9.5M)

    # Stats
    total_points: int
    points_per_game: float = 0.0
    selected_by_percent: float = 0.0
    form: float = 0.0
    minutes: int = 0
    goals_scored: int = 0
    assists: int = 0
    clean_sheets: int = 0
    goals_conceded: int = 0
    bonus: int = 0

    # Expected stats
    expected_goals: float = Field(default=0.0, alias='expected_goals')
    expected_assists: float = Field(default=0.0, alias='expected_assists')
    expected_goal_involvements: float = Field(default=0.0, alias='expected_goal_involvements')
    expected_goals_conceded: float = Field(default=0.0, alias='expected_goals_conceded')

    # Availability
    chance_of_playing_next_round: Optional[int] = None
    news: str = ""
    status: str = "a"  # a=available, d=doubtful, i=injured, u=unavailable

    class Config:
        populate_by_name = True
        extra = 'ignore'

    @computed_field
    @property
    def position_name(self) -> str:
        """Get readable position name."""
        position_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        return position_map.get(self.position, 'UNK')

    @computed_field
    @property
    def cost_millions(self) -> float:
        """Get cost in millions."""
        return self.cost / 10

    @computed_field
    @property
    def is_available(self) -> bool:
        """Check if player is available for selection."""
        if self.status != 'a':
            return False
        if self.chance_of_playing_next_round is not None:
            return self.chance_of_playing_next_round >= 50
        return True

    @computed_field
    @property
    def value(self) -> float:
        """Calculate points per million value."""
        if self.cost <= 0:
            return 0.0
        return (self.total_points / self.cost) * 10


class Team(BaseModel):
    """Represents an FPL team."""

    id: int
    name: str
    short_name: str
    strength: int
    strength_overall_home: int
    strength_overall_away: int
    strength_attack_home: int
    strength_attack_away: int
    strength_defence_home: int
    strength_defence_away: int

    class Config:
        extra = 'ignore'


class Fixture(BaseModel):
    """Represents a fixture."""

    id: int
    event: int  # Gameweek number
    team_h: int  # Home team ID
    team_a: int  # Away team ID
    team_h_difficulty: int
    team_a_difficulty: int
    kickoff_time: Optional[str] = None
    started: bool = False
    finished: bool = False
    finished_provisional: bool = False

    # Score (if finished)
    team_h_score: Optional[int] = None
    team_a_score: Optional[int] = None

    class Config:
        extra = 'ignore'


class GameWeek(BaseModel):
    """Represents a gameweek."""

    id: int
    name: str
    deadline_time: str
    is_previous: bool = False
    is_current: bool = False
    is_next: bool = False
    finished: bool = False

    class Config:
        extra = 'ignore'


class PlayerHistory(BaseModel):
    """Player performance history for a single gameweek."""

    element: int  # Player ID
    fixture: int  # Fixture ID
    opponent_team: int
    total_points: int
    was_home: bool
    kickoff_time: str
    round: int  # Gameweek
    minutes: int
    goals_scored: int
    assists: int
    clean_sheets: int
    goals_conceded: int
    bonus: int
    bps: int  # Bonus points system score
    value: int  # Player cost at time of fixture

    class Config:
        extra = 'ignore'


class Squad(BaseModel):
    """Represents a user's FPL squad."""

    picks: List[int]  # List of player IDs
    captain_id: int
    vice_captain_id: int
    formation: str = "4-4-2"
    total_cost: float = 0.0
    expected_points: float = 0.0

    class Config:
        extra = 'ignore'

    def get_starting_11_ids(self) -> List[int]:
        """Get the starting 11 player IDs."""
        return self.picks[:11] if len(self.picks) >= 11 else self.picks

    def get_bench_ids(self) -> List[int]:
        """Get the bench player IDs."""
        return self.picks[11:] if len(self.picks) > 11 else []


class Transfer(BaseModel):
    """Represents a player transfer."""

    player_in_id: int
    player_out_id: int
    player_in_cost: float
    player_out_cost: float
    points_gain: float = 0.0
    priority: int = 1  # Priority order for multiple transfers

    @computed_field
    @property
    def cost_delta(self) -> float:
        """Calculate cost difference."""
        return self.player_in_cost - self.player_out_cost


class SquadConstraints(BaseModel):
    """Constraints for building an FPL squad."""

    total_budget: float = 100.0  # Budget in millions
    squad_size: int = 15
    starting_11: int = 11
    max_players_per_team: int = 3

    # Position requirements for full squad
    num_goalkeepers: int = 2
    num_defenders: int = 5
    num_midfielders: int = 5
    num_forwards: int = 3

    # Starting lineup position constraints
    min_starting_goalkeepers: int = 1
    max_starting_goalkeepers: int = 1
    min_starting_defenders: int = 3
    max_starting_defenders: int = 5
    min_starting_midfielders: int = 2
    max_starting_midfielders: int = 5
    min_starting_forwards: int = 1
    max_starting_forwards: int = 3


class OptimizedSquad(BaseModel):
    """Represents an optimized FPL squad with metadata."""

    players: List[Player]
    starting_11_ids: List[int]
    captain_id: int
    vice_captain_id: int
    total_cost: float
    expected_points: float
    formation: str

    class Config:
        arbitrary_types_allowed = True

    def get_starting_players(self) -> List[Player]:
        """Get the starting 11 players."""
        return [p for p in self.players if p.id in self.starting_11_ids]

    def get_bench_players(self) -> List[Player]:
        """Get the bench players."""
        return [p for p in self.players if p.id not in self.starting_11_ids]

    def get_players_by_position(self, position: str) -> List[Player]:
        """Get players filtered by position name."""
        return [p for p in self.players if p.position_name == position]


class TransferRecommendation(BaseModel):
    """Represents a transfer recommendation."""

    transfers: List[Transfer]
    expected_points_gain: float
    total_cost: int  # In points (-4 per transfer)
    net_gain: float  # Expected points gain minus transfer cost
    squad_after: OptimizedSquad

    class Config:
        arbitrary_types_allowed = True

    @computed_field
    @property
    def num_transfers(self) -> int:
        """Number of transfers."""
        return len(self.transfers)