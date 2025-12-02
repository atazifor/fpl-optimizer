"""FastAPI server for FPL optimizer."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os

from .fpl_client import FPLClient
from .models import Player, Team, Fixture, SquadConstraints
from .optimizer import FPLOptimizer, OptimizationConfig
from .predictor import ExpectedPointsPredictor

app = FastAPI(title="FPL Optimizer API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],  # Vite ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class PlayerResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    name: str
    team_id: int
    position_name: str
    cost: float
    cost_millions: float
    total_points: int
    form: float
    selected_by_percent: float
    is_available: bool
    expected_points: Optional[float] = None


class TransferRecommendation(BaseModel):
    player_out_id: int
    player_out_name: str
    player_in_id: int
    player_in_name: str
    cost_delta: float
    points_gain: float


class OptimizedSquadResponse(BaseModel):
    players: List[PlayerResponse]
    starting_11_ids: List[int]
    captain_id: int
    vice_captain_id: int
    total_cost: float
    expected_points: float
    formation: str


class ChipRecommendation(BaseModel):
    chip_name: str
    recommended_gameweek: Optional[int]
    reasoning: str
    priority: int  # 1 = highest priority


class MyTeamResponse(BaseModel):
    team_name: str
    team_value: float
    bank: float
    free_transfers: int
    current_gameweek: int
    chips_available: List[str]
    chip_recommendations: List[ChipRecommendation]
    players: List[PlayerResponse]
    starting_11_ids: List[int]
    captain_id: int
    vice_captain_id: int


class CaptainOption(BaseModel):
    player_id: int
    player_name: str
    team_name: str
    expected_points: float
    fixture: str


class TransferSuggestion(BaseModel):
    player_out_id: int
    player_out_name: str
    player_out_position: str
    player_out_cost: float
    player_out_fixtures: List[str]  # Next 3 fixtures
    player_in_id: int
    player_in_name: str
    player_in_position: str
    player_in_cost: float
    player_in_fixtures: List[str]  # Next 3 fixtures
    cost_change: float


class TransferRecommendationResponse(BaseModel):
    free_transfers: int
    num_transfers: int
    transfer_cost: int
    expected_gain: float
    net_gain: float
    recommendation: str  # "make_transfers", "wait", "not_worth_it"
    transfers: List[TransferSuggestion]
    captain_pick: Optional[CaptainOption] = None
    vice_captain_pick: Optional[CaptainOption] = None


# Global client (reuse across requests)
client = None


def get_client():
    """Get or create FPL client."""
    global client
    if client is None:
        client = FPLClient()
    return client


@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "message": "FPL Optimizer API"}


@app.get("/api/players", response_model=List[PlayerResponse])
async def get_players(available_only: bool = True):
    """Get all players."""
    try:
        fpl_client = get_client()
        bootstrap = fpl_client.get_bootstrap_static()

        players = [Player(**p) for p in bootstrap['elements']]

        if available_only:
            players = [p for p in players if p.is_available]

        return [
            PlayerResponse(
                id=p.id,
                name=p.name,
                team_id=p.team_id,
                position_name=p.position_name,
                cost=p.cost,
                cost_millions=p.cost_millions,
                total_points=p.total_points,
                form=p.form,
                selected_by_percent=p.selected_by_percent,
                is_available=p.is_available
            )
            for p in players
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/my-team", response_model=MyTeamResponse)
async def get_my_team():
    """Get current team details."""
    try:
        fpl_client = get_client()

        # Fetch data
        my_team = fpl_client.get_my_team()
        my_info = fpl_client.get_my_info()
        my_history = fpl_client.get_my_history()
        bootstrap = fpl_client.get_bootstrap_static()

        # Get player IDs and captain info
        current_player_ids = [pick['element'] for pick in my_team['picks']]
        starting_11_ids = [pick['element'] for pick in my_team['picks'][:11]]

        # Extract captain and vice-captain IDs
        captain_id = next((pick['element'] for pick in my_team['picks'] if pick.get('is_captain')), 0)
        vice_captain_id = next((pick['element'] for pick in my_team['picks'] if pick.get('is_vice_captain')), 0)

        # Get players
        all_players = [Player(**p) for p in bootstrap['elements']]
        squad_players = [p for p in all_players if p.id in current_player_ids]

        # Get team stats
        bank = my_team['entry_history']['bank'] / 10
        team_value = my_team['entry_history']['value'] / 10

        # Calculate free transfers for next gameweek
        # Logic: You get 1 FT per GW, can bank 1 (max 2 total)
        # Need to look at last 2 GWs to determine banking
        current_gws = my_history['current']

        if len(current_gws) >= 2:
            last_gw = current_gws[-1]
            prev_gw = current_gws[-2]

            # Calculate FT available in last GW based on previous GW
            prev_transfers = prev_gw.get('event_transfers', 0)
            prev_cost = prev_gw.get('event_transfers_cost', 0)

            # If no transfers in prev GW, they banked (had 2 FT in last GW)
            # Otherwise they had 1 FT in last GW
            ft_available_last_gw = 2 if prev_transfers == 0 else 1

            # Calculate for next GW based on last GW usage
            last_transfers = last_gw.get('event_transfers', 0)
            last_cost = last_gw.get('event_transfers_cost', 0)

            # If no transfers, bank it (will have 2 for next)
            if last_transfers == 0:
                free_transfers = min(2, ft_available_last_gw + 1)
            # If used some but not all FT, bank the rest
            elif last_cost == 0:  # No hits taken
                free_transfers = min(2, (ft_available_last_gw - last_transfers) + 1)
            # If took hits, reset to 1
            else:
                free_transfers = 1
        else:
            # Default to 1 if not enough history
            free_transfers = 1

        # Get current gameweek
        current_gw = fpl_client.get_current_gameweek()

        # Get chips
        chips_used = my_history.get('chips', [])
        chips_available = []
        if not any(c['name'] == 'wildcard' for c in chips_used):
            chips_available.append('Wildcard')
        if not any(c['name'] == 'freehit' for c in chips_used):
            chips_available.append('Free Hit')
        if not any(c['name'] == 'bboost' for c in chips_used):
            chips_available.append('Bench Boost')
        if not any(c['name'] == '3xc' for c in chips_used):
            chips_available.append('Triple Captain')

        # Generate chip recommendations
        # Note: All chips reset at GW20 (mid-season), so you get them twice per season
        chip_recommendations = []

        # Determine which half of season we're in
        is_first_half = current_gw <= 19

        # Wildcard - recommend around GW16-18 (first half) or GW34-36 (second half)
        if 'Wildcard' in chips_available:
            if is_first_half:
                if current_gw < 16:
                    chip_recommendations.append(ChipRecommendation(
                        chip_name="Wildcard",
                        recommended_gameweek=17,
                        reasoning="Use during winter fixture congestion to overhaul your squad (resets at GW20)",
                        priority=2
                    ))
            else:
                if current_gw < 34:
                    chip_recommendations.append(ChipRecommendation(
                        chip_name="Wildcard",
                        recommended_gameweek=34,
                        reasoning="Use before DGW to prepare for Bench Boost and maximize fixture coverage",
                        priority=1
                    ))

        # Free Hit - recommend for blank/double gameweeks
        if 'Free Hit' in chips_available:
            if is_first_half:
                chip_recommendations.append(ChipRecommendation(
                    chip_name="Free Hit",
                    recommended_gameweek=18,
                    reasoning="Use on a blank or difficult gameweek (resets at GW20)",
                    priority=3
                ))
            else:
                # Typically GW25 (blank) or DGWs around GW32-37
                if current_gw < 25:
                    chip_recommendations.append(ChipRecommendation(
                        chip_name="Free Hit",
                        recommended_gameweek=25,
                        reasoning="Use on blank gameweek when many teams don't play",
                        priority=3
                    ))
                else:
                    chip_recommendations.append(ChipRecommendation(
                        chip_name="Free Hit",
                        recommended_gameweek=None,
                        reasoning="Save for an upcoming blank or double gameweek",
                        priority=3
                    ))

        # Bench Boost - recommend for best fixture runs or DGWs
        if 'Bench Boost' in chips_available:
            if is_first_half:
                # In first half, look for good fixture runs (typically around GW16-18)
                chip_recommendations.append(ChipRecommendation(
                    chip_name="Bench Boost",
                    recommended_gameweek=17,
                    reasoning="Use when your full squad has favorable fixtures (resets at GW20)",
                    priority=3
                ))
            else:
                # In second half, target DGWs
                if current_gw < 32:
                    chip_recommendations.append(ChipRecommendation(
                        chip_name="Bench Boost",
                        recommended_gameweek=36,
                        reasoning="Use on a double gameweek when all 15 players have good fixtures",
                        priority=1
                    ))
                else:
                    chip_recommendations.append(ChipRecommendation(
                        chip_name="Bench Boost",
                        recommended_gameweek=37,
                        reasoning="Use on remaining double gameweek with full team coverage",
                        priority=1
                    ))

        # Triple Captain - recommend for premium captains with good fixtures
        if 'Triple Captain' in chips_available:
            if is_first_half:
                # In first half, use on best captain fixture
                chip_recommendations.append(ChipRecommendation(
                    chip_name="Triple Captain",
                    recommended_gameweek=18,
                    reasoning="Use when your premium player has an excellent fixture (resets at GW20)",
                    priority=3
                ))
            else:
                # In second half, target DGWs
                if current_gw < 32:
                    chip_recommendations.append(ChipRecommendation(
                        chip_name="Triple Captain",
                        recommended_gameweek=36,
                        reasoning="Use on a DGW when your premium player has two great fixtures",
                        priority=2
                    ))
                else:
                    chip_recommendations.append(ChipRecommendation(
                        chip_name="Triple Captain",
                        recommended_gameweek=None,
                        reasoning="Use when your captain has a double gameweek",
                        priority=2
                    ))

        # Sort by priority
        chip_recommendations.sort(key=lambda x: x.priority)

        return MyTeamResponse(
            team_name=my_info['name'],
            team_value=team_value,
            bank=bank,
            free_transfers=free_transfers,
            current_gameweek=current_gw,
            chips_available=chips_available,
            chip_recommendations=chip_recommendations,
            players=[
                PlayerResponse(
                    id=p.id,
                    name=p.name,
                    team_id=p.team_id,
                    position_name=p.position_name,
                    cost=p.cost,
                    cost_millions=p.cost_millions,
                    total_points=p.total_points,
                    form=p.form,
                    selected_by_percent=p.selected_by_percent,
                    is_available=p.is_available
                )
                for p in squad_players
            ],
            starting_11_ids=starting_11_ids,
            captain_id=captain_id,
            vice_captain_id=vice_captain_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/optimize", response_model=OptimizedSquadResponse)
async def optimize_squad(budget: float = 100.0, objective: str = "points"):
    """Optimize squad selection."""
    try:
        fpl_client = get_client()
        bootstrap = fpl_client.get_bootstrap_static()
        fixtures_data = fpl_client.get_fixtures()

        players = [Player(**p) for p in bootstrap['elements']]
        available_players = [p for p in players if p.is_available]
        teams_data = [Team(**t) for t in bootstrap['teams']]
        teams = {t.id: t for t in teams_data}
        fixtures = [Fixture(**f) for f in fixtures_data]

        constraints = SquadConstraints(total_budget=budget)
        optimizer = FPLOptimizer(available_players, fixtures, teams, constraints)
        squad = optimizer.optimize_squad()

        return OptimizedSquadResponse(
            players=[
                PlayerResponse(
                    id=p.id,
                    name=p.name,
                    team_id=p.team_id,
                    position_name=p.position_name,
                    cost=p.cost,
                    cost_millions=p.cost_millions,
                    total_points=p.total_points,
                    form=p.form,
                    selected_by_percent=p.selected_by_percent,
                    is_available=p.is_available
                )
                for p in squad.players
            ],
            starting_11_ids=squad.starting_11_ids,
            captain_id=squad.captain_id,
            vice_captain_id=squad.vice_captain_id,
            total_cost=squad.total_cost / 10,  # Convert to millions
            expected_points=squad.expected_points,
            formation=squad.formation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/transfer-recommendations", response_model=TransferRecommendationResponse)
async def get_transfer_recommendations(num_transfers: Optional[int] = None):
    """Get transfer recommendations with hit analysis."""
    try:
        fpl_client = get_client()

        # Fetch data
        my_team = fpl_client.get_my_team()
        my_history = fpl_client.get_my_history()
        bootstrap = fpl_client.get_bootstrap_static()

        # Calculate free transfers (same logic as my-team endpoint)
        current_gws = my_history['current']
        if len(current_gws) >= 2:
            last_gw = current_gws[-1]
            prev_gw = current_gws[-2]
            prev_transfers = prev_gw.get('event_transfers', 0)
            ft_available_last_gw = 2 if prev_transfers == 0 else 1
            last_transfers = last_gw.get('event_transfers', 0)
            last_cost = last_gw.get('event_transfers_cost', 0)

            if last_transfers == 0:
                free_transfers = min(2, ft_available_last_gw + 1)
            elif last_cost == 0:
                free_transfers = min(2, (ft_available_last_gw - last_transfers) + 1)
            else:
                free_transfers = 1
        else:
            free_transfers = 1

        # Get current squad
        current_player_ids = [pick['element'] for pick in my_team['picks']]

        # Get all players
        all_players = [Player(**p) for p in bootstrap['elements']]
        players_dict = {p.id: p for p in all_players}

        # Get fixtures and teams for xP calculation
        current_gw = fpl_client.get_current_gameweek()
        fixtures_data = fpl_client.get_fixtures()
        teams_data = [Team(**t) for t in bootstrap['teams']]
        teams = {t.id: t for t in teams_data}
        fixtures = [Fixture(**f) for f in fixtures_data]

        # Run new optimizer with multi-GW planning
        config = OptimizationConfig(
            horizon_weeks=2,  # Look ahead 2 weeks
            free_transfers=free_transfers,
            max_transfers=num_transfers  # Force specific number of transfers if provided
        )
        optimizer = FPLOptimizer(all_players, fixtures, teams)
        result = optimizer.optimize_with_transfers(
            current_player_ids,
            config=config
        )

        # Extract first week's transfer plan
        first_plan = result.transfer_plans[0] if result.transfer_plans else None

        if first_plan:
            players_out_ids = set(first_plan.transfers_out)
            players_in_ids = set(first_plan.transfers_in)
            actual_transfers = len(players_out_ids)
            # Recalculate transfer cost to ensure correctness
            hits = max(0, actual_transfers - free_transfers)
            transfer_cost = hits * 4
        else:
            # No transfers recommended
            players_out_ids = set()
            players_in_ids = set()
            actual_transfers = 0
            transfer_cost = 0

        new_squad = result.current_squad

        # Calculate expected gain (compare xP for next GW only)
        # Get current squad's expected points for next gameweek
        next_gw = current_gw + 1
        current_players = [p for p in all_players if p.id in current_player_ids]

        # Calculate current squad's xP for next gameweek
        current_squad_xp = 0
        for p in current_players[:11]:  # Starting 11
            xp = optimizer._get_xp(p.id, next_gw)
            current_squad_xp += xp

        # Add captain bonus for current squad (assume best player is captain)
        current_squad_sorted = sorted(
            [(p, optimizer._get_xp(p.id, next_gw)) for p in current_players[:11]],
            key=lambda x: x[1],
            reverse=True
        )
        if current_squad_sorted:
            current_squad_xp += current_squad_sorted[0][1]  # Captain gets 2x (add 1x more)

        # Compare with new squad's expected points (already includes captain bonus)
        expected_gain = new_squad.expected_points - current_squad_xp
        net_gain = expected_gain - transfer_cost

        # Determine recommendation
        if net_gain > 2:
            recommendation = "make_transfers"
        elif net_gain >= 0:
            recommendation = "wait"
        else:
            recommendation = "not_worth_it"

        # Helper function to get next 3 fixtures for a player
        def get_player_fixtures(player_id: int, team_id: int) -> List[str]:
            next_3_gws = [current_gw + 1, current_gw + 2, current_gw + 3]
            fixture_list = []

            for gw in next_3_gws:
                player_fixtures = [
                    f for f in fixtures
                    if (f.team_h == team_id or f.team_a == team_id) and f.event == gw
                ]

                if player_fixtures:
                    fixture = player_fixtures[0]
                    if fixture.team_h == team_id:
                        opponent = teams[fixture.team_a].short_name
                        difficulty = fixture.team_h_difficulty
                        fixture_list.append(f"vs {opponent} ({difficulty})")
                    else:
                        opponent = teams[fixture.team_h].short_name
                        difficulty = fixture.team_a_difficulty
                        fixture_list.append(f"@ {opponent} ({difficulty})")
                else:
                    fixture_list.append("Blank")

            return fixture_list

        # Build transfer suggestions
        transfers = []
        for out_id, in_id in zip(sorted(players_out_ids), sorted(players_in_ids)):
            player_out = players_dict[out_id]
            player_in = players_dict[in_id]

            # Get fixtures for both players
            player_out_fixtures = get_player_fixtures(player_out.id, player_out.team_id)
            player_in_fixtures = get_player_fixtures(player_in.id, player_in.team_id)

            transfers.append(
                TransferSuggestion(
                    player_out_id=out_id,
                    player_out_name=player_out.name,
                    player_out_position=player_out.position_name,
                    player_out_cost=player_out.cost_millions,
                    player_out_fixtures=player_out_fixtures,
                    player_in_id=in_id,
                    player_in_name=player_in.name,
                    player_in_position=player_in.position_name,
                    player_in_cost=player_in.cost_millions,
                    player_in_fixtures=player_in_fixtures,
                    cost_change=player_in.cost_millions - player_out.cost_millions
                )
            )

        # Calculate captain/vice-captain picks for next GW
        next_gw = current_gw + 1

        # Get squad after transfers (or current if no transfers)
        new_set = set(p.id for p in new_squad.players)
        final_squad_ids = list(new_set) if actual_transfers > 0 else current_player_ids
        final_squad_players = [p for p in all_players if p.id in final_squad_ids]

        # Create predictor
        predictor = ExpectedPointsPredictor(all_players, teams, fixtures)

        # Calculate captain scores for NEXT gameweek
        captain_scores = []
        for player in final_squad_players[:11]:  # Only starting 11
            expected_pts = predictor.calculate_expected_points(player, next_gw, num_gameweeks=1)
            captain_value = expected_pts * 2

            # Get fixture for NEXT gameweek
            player_fixtures = [
                f for f in fixtures
                if (f.team_h == player.team_id or f.team_a == player.team_id)
                and f.event == next_gw
            ]

            fixture_str = ""
            if player_fixtures:
                fixture = player_fixtures[0]
                if fixture.team_h == player.team_id:
                    opponent = teams[fixture.team_a].short_name
                    fixture_str = f"vs {opponent} (H)"
                else:
                    opponent = teams[fixture.team_h].short_name
                    fixture_str = f"@ {opponent} (A)"

            captain_scores.append(
                CaptainOption(
                    player_id=player.id,
                    player_name=player.name,
                    team_name=teams[player.team_id].short_name,
                    expected_points=captain_value,
                    fixture=fixture_str
                )
            )

        # Sort by expected points
        captain_scores.sort(key=lambda x: x.expected_points, reverse=True)
        captain_pick = captain_scores[0] if len(captain_scores) > 0 else None
        vice_captain_pick = captain_scores[1] if len(captain_scores) > 1 else None

        return TransferRecommendationResponse(
            free_transfers=free_transfers,
            num_transfers=actual_transfers,
            transfer_cost=transfer_cost,
            expected_gain=expected_gain,
            net_gain=net_gain,
            recommendation=recommendation,
            transfers=transfers,
            captain_pick=captain_pick,
            vice_captain_pick=vice_captain_pick
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/captain-picks", response_model=List[CaptainOption])
async def get_captain_picks():
    """Get top 5 captain recommendations for NEXT gameweek."""
    try:
        fpl_client = get_client()

        # Fetch data
        bootstrap = fpl_client.get_bootstrap_static()
        fixtures_data = fpl_client.get_fixtures()
        my_team = fpl_client.get_my_team()
        current_gw = fpl_client.get_current_gameweek()

        # Captain picks are for NEXT gameweek
        next_gw = current_gw + 1

        # Parse data
        players = [Player(**p) for p in bootstrap['elements']]
        teams_data = [Team(**t) for t in bootstrap['teams']]
        teams = {t.id: t for t in teams_data}
        fixtures = [Fixture(**f) for f in fixtures_data]

        # Get squad players
        current_player_ids = [pick['element'] for pick in my_team['picks'][:11]]
        squad_players = [p for p in players if p.id in current_player_ids]

        # Create predictor
        predictor = ExpectedPointsPredictor(players, teams, fixtures)

        # Calculate captain scores for NEXT gameweek
        captain_scores = []
        for player in squad_players:
            expected_pts = predictor.calculate_expected_points(player, next_gw, num_gameweeks=1)
            captain_value = expected_pts * 2

            # Get fixture for NEXT gameweek
            player_fixtures = [
                f for f in fixtures
                if (f.team_h == player.team_id or f.team_a == player.team_id)
                and f.event == next_gw
            ]

            fixture_str = ""
            if player_fixtures:
                fixture = player_fixtures[0]
                if fixture.team_h == player.team_id:
                    opponent = teams[fixture.team_a].short_name
                    fixture_str = f"vs {opponent} (H)"
                else:
                    opponent = teams[fixture.team_h].short_name
                    fixture_str = f"@ {opponent} (A)"

            captain_scores.append(
                CaptainOption(
                    player_id=player.id,
                    player_name=player.name,
                    team_name=teams[player.team_id].short_name,
                    expected_points=captain_value,
                    fixture=fixture_str
                )
            )

        # Sort and return top 5
        captain_scores.sort(key=lambda x: x.expected_points, reverse=True)
        return captain_scores[:5]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class InjuryInfo(BaseModel):
    player_id: int
    player_name: str
    team_name: str
    position: str
    injury_status: str  # "Injured", "Doubtful", "Suspended"
    news: str  # Raw news text from FPL API
    expected_return: Optional[str] = None
    selected_by_percent: float
    cost: float


@app.get("/api/injuries", response_model=List[InjuryInfo])
async def get_injuries():
    """Get high-profile injuries (players with >5% ownership or >£7.0M cost)."""
    try:
        fpl_client = get_client()
        bootstrap = fpl_client.get_bootstrap_static()

        # Parse data
        players = [Player(**p) for p in bootstrap['elements']]
        teams_data = [Team(**t) for t in bootstrap['teams']]
        teams = {t.id: t for t in teams_data}

        # Filter injured/suspended players
        injured_players = []
        for player in players:
            # Check if player is unavailable
            if not player.is_available:
                # Only include high-profile players (>5% ownership OR >£7.0M)
                if player.selected_by_percent >= 5.0 or player.cost_millions >= 7.0:
                    # Determine status
                    if player.chance_of_playing_next_round is not None and player.chance_of_playing_next_round < 100:
                        if player.chance_of_playing_next_round == 0:
                            status = "Injured"
                        elif player.chance_of_playing_next_round <= 25:
                            status = "Doubtful (25%)"
                        elif player.chance_of_playing_next_round <= 50:
                            status = "Doubtful (50%)"
                        elif player.chance_of_playing_next_round <= 75:
                            status = "75% chance"
                        else:
                            status = "Doubtful"
                    else:
                        status = "Unavailable"

                    injured_players.append(
                        InjuryInfo(
                            player_id=player.id,
                            player_name=player.name,
                            team_name=teams[player.team_id].short_name,
                            position=player.position_name,
                            injury_status=status,
                            news=player.news if player.news else "",
                            expected_return=None,  # FPL API doesn't provide this reliably
                            selected_by_percent=player.selected_by_percent,
                            cost=player.cost_millions
                        )
                    )

        # Sort by ownership (most owned first)
        injured_players.sort(key=lambda x: x.selected_by_percent, reverse=True)
        return injured_players

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class TeamFixture(BaseModel):
    team_id: int
    team_name: str
    fixtures: List[str]  # e.g., ["vs ARS (H, 4)", "@ LIV (A, 5)"]
    avg_difficulty: float
    total_difficulty: int


@app.get("/api/fixtures", response_model=List[TeamFixture])
async def get_fixtures():
    """Get next 3 gameweeks fixtures with difficulty ratings for all teams."""
    try:
        fpl_client = get_client()
        bootstrap = fpl_client.get_bootstrap_static()
        fixtures_data = fpl_client.get_fixtures()
        current_gw = fpl_client.get_current_gameweek()

        # Parse data
        teams_data = [Team(**t) for t in bootstrap['teams']]
        teams = {t.id: t for t in teams_data}
        fixtures = [Fixture(**f) for f in fixtures_data]

        # Get next 3 gameweeks
        next_3_gws = [current_gw + 1, current_gw + 2, current_gw + 3]

        # Build fixture summary for each team
        team_fixtures = []
        for team in teams.values():
            team_fixture_list = []
            difficulties = []

            for gw in next_3_gws:
                gw_fixtures = [
                    f for f in fixtures
                    if (f.team_h == team.id or f.team_a == team.id) and f.event == gw
                ]

                if gw_fixtures:
                    for fixture in gw_fixtures:
                        if fixture.team_h == team.id:
                            opponent = teams[fixture.team_a].short_name
                            difficulty = fixture.team_h_difficulty
                            team_fixture_list.append(f"vs {opponent} (H, {difficulty})")
                            difficulties.append(difficulty)
                        else:
                            opponent = teams[fixture.team_h].short_name
                            difficulty = fixture.team_a_difficulty
                            team_fixture_list.append(f"@ {opponent} (A, {difficulty})")
                            difficulties.append(difficulty)
                else:
                    team_fixture_list.append("Blank")

            avg_diff = sum(difficulties) / len(difficulties) if difficulties else 0
            total_diff = sum(difficulties)

            team_fixtures.append(
                TeamFixture(
                    team_id=team.id,
                    team_name=team.name,
                    fixtures=team_fixture_list,
                    avg_difficulty=round(avg_diff, 2),
                    total_difficulty=total_diff
                )
            )

        # Sort by avg difficulty (easiest first)
        team_fixtures.sort(key=lambda x: x.avg_difficulty)
        return team_fixtures

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)