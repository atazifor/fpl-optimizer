"""FastAPI server for FPL optimizer."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .fpl_client import FPLClient
from .models import Player, Team, Fixture, SquadConstraints
from .optimizer import FPLOptimizer, OptimizationConfig, Chip
from .expected_points import ExpectedPointsPredictor, PlayerStats, TeamStrength  # Compatibility wrapper
from .data_cache import DataCache

logger = logging.getLogger(__name__)

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
    player_out_team: str
    player_out_cost: float
    player_out_total_points: int
    player_out_form: float
    player_out_expected_points: float
    player_out_fixtures: List[str]  # Next 3 fixtures
    player_in_id: int
    player_in_name: str
    player_in_position: str
    player_in_team: str
    player_in_cost: float
    player_in_total_points: int
    player_in_form: float
    player_in_expected_points: float
    player_in_fixtures: List[str]  # Next 3 fixtures
    cost_change: float
    points_gain: float


class TransferRecommendationResponse(BaseModel):
    free_transfers: int
    num_transfers: int
    transfer_cost: int
    expected_gain: float  # Next gameweek only
    net_gain: float  # Next gameweek only (expected_gain - transfer_cost)
    horizon_expected_gain: float  # Total over planning horizon
    horizon_net_gain: float  # Total over planning horizon (horizon_expected_gain - transfer_cost)
    planning_horizon_weeks: int  # Number of weeks planned ahead
    recommendation: str  # "make_transfers", "wait", "not_worth_it"
    reasoning: str  # Why this plan was chosen (e.g., "Sets up for GW17 Bench Boost")
    transfers: List[TransferSuggestion]
    captain_pick: Optional[CaptainOption] = None
    vice_captain_pick: Optional[CaptainOption] = None
    bench_order: List[int] = []  # Player IDs in bench order (12th, 13th, 14th, 15th)


# Global client (reuse across requests)
client = None
# Global cache for advanced stats
cached_player_stats: Optional[Dict[int, PlayerStats]] = None
cached_team_strength: Optional[Dict[int, TeamStrength]] = None
cache_loaded = False


def get_client():
    """Get or create FPL client."""
    global client
    if client is None:
        client = FPLClient()
    return client


def load_advanced_stats() -> tuple[Optional[Dict[int, PlayerStats]], Optional[Dict[int, TeamStrength]]]:
    """
    Load advanced stats from cache if available.

    Returns cached player stats and team strength, or (None, None) if not available.
    Uses FPL_ADVANCED_MODE env variable to enable/disable (default: enabled).
    """
    global cached_player_stats, cached_team_strength, cache_loaded

    # Check if already loaded
    if cache_loaded:
        return cached_player_stats, cached_team_strength

    # Check if advanced mode is enabled (default: yes)
    advanced_mode_enabled = os.getenv('FPL_ADVANCED_MODE', 'true').lower() == 'true'

    if not advanced_mode_enabled:
        logger.info("Advanced mode disabled via FPL_ADVANCED_MODE=false")
        cache_loaded = True
        return None, None

    try:
        cache = DataCache()
        cached_player_stats = cache.load_player_stats()
        cached_team_strength = cache.load_team_strength()

        if cached_player_stats and cached_team_strength:
            logger.info(f"Loaded advanced stats from cache: {len(cached_player_stats)} players, {len(cached_team_strength)} teams")
        else:
            logger.info("No valid cache found - using FPL data only")
            logger.info("Run 'python scripts/fetch_external_data.py' to enable advanced predictions")

        cache_loaded = True
        return cached_player_stats, cached_team_strength

    except Exception as e:
        logger.error(f"Failed to load advanced stats cache: {e}")
        cache_loaded = True
        return None, None


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
        my_info = fpl_client.get_my_info()
        my_history = fpl_client.get_my_history()
        bootstrap = fpl_client.get_bootstrap_static()
        current_gw = fpl_client.get_current_gameweek()

        # Check if Free Hit was used in the current or any recent gameweek
        chips_used = my_history.get('chips', [])

        # Find the most recent Free Hit usage
        free_hit_gw = None
        for chip in chips_used:
            if chip['name'] == 'freehit':
                free_hit_gw = chip['event']

        # Check if current GW is finished
        events = bootstrap.get('events', [])
        current_gw_data = next((e for e in events if e['id'] == current_gw), None)
        is_gw_finished = current_gw_data and current_gw_data.get('finished', False)

        # Determine which GW to fetch team data from
        # After Free Hit GW ends, the team reverts to the state from the GW before Free Hit
        # But we can't fetch future GW data, so:
        # - If Free Hit was used in current GW and it's finished, fetch from GW before Free Hit
        # - Otherwise, fetch current GW
        if free_hit_gw == current_gw and is_gw_finished:
            # Free Hit was used this GW and it's finished - team reverted to previous permanent state
            # Fetch from the gameweek before Free Hit
            my_team = fpl_client.get_my_team(gameweek=current_gw - 1)
        else:
            # Normal case: fetch current gameweek
            my_team = fpl_client.get_my_team()

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


@app.get("/api/next-gw-team", response_model=MyTeamResponse)
async def get_next_gw_team(num_transfers: Optional[int] = None):
    """Get projected team for NEXT gameweek after applying recommended transfers."""
    try:
        fpl_client = get_client()

        # Fetch basic data
        my_info = fpl_client.get_my_info()
        my_history = fpl_client.get_my_history()
        bootstrap = fpl_client.get_bootstrap_static()

        # Check if Free Hit was used in the current gameweek
        chips_used = my_history.get('chips', [])
        free_hit_gw = None
        for chip in chips_used:
            if chip['name'] == 'freehit':
                free_hit_gw = chip['event']

        # Get current gameweek and check if it's finished
        events = bootstrap.get('events', [])
        current_gw = next((e['id'] for e in events if e['is_current']), None)
        current_gw_data = next((e for e in events if e['id'] == current_gw), None)
        is_gw_finished = current_gw_data and current_gw_data.get('finished', False)

        # Fetch correct team data (reverted team if Free Hit was used in finished GW)
        if free_hit_gw == current_gw and is_gw_finished:
            my_team = fpl_client.get_my_team(gameweek=current_gw - 1)
        else:
            my_team = fpl_client.get_my_team()

        # Calculate free transfers
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
        current_squad_players = [p for p in all_players if p.id in current_player_ids]
        available_for_transfer = [p for p in all_players if p.is_available]

        # Combine for optimization
        all_player_ids = set([p.id for p in current_squad_players] + [p.id for p in available_for_transfer])
        optimization_players = [p for p in all_players if p.id in all_player_ids]

        # Get fixtures and teams
        current_gw = fpl_client.get_current_gameweek()
        fixtures_data = fpl_client.get_fixtures()
        teams_data = [Team(**t) for t in bootstrap['teams']]
        teams = {t.id: t for t in teams_data}
        fixtures = [Fixture(**f) for f in fixtures_data]

        # Get available chips
        chips_used = my_history.get('chips', [])
        available_chips = []

        if not any(c['name'] == 'wildcard' for c in chips_used):
            available_chips.append(Chip.WILDCARD)
        if not any(c['name'] == 'freehit' for c in chips_used):
            available_chips.append(Chip.FREE_HIT)
        if not any(c['name'] == 'bboost' for c in chips_used):
            available_chips.append(Chip.BENCH_BOOST)
        if not any(c['name'] == '3xc' for c in chips_used):
            available_chips.append(Chip.TRIPLE_CAPTAIN)

        # Determine horizon based on chip timing
        horizon_weeks = 2
        chip_gws = []

        if Chip.BENCH_BOOST in available_chips:
            bb_gw = 17 if current_gw <= 19 else 36
            if bb_gw > current_gw:
                chip_gws.append(bb_gw)

        if Chip.TRIPLE_CAPTAIN in available_chips:
            tc_gw = 18 if current_gw <= 19 else 36
            if tc_gw > current_gw:
                chip_gws.append(tc_gw)

        if chip_gws:
            max_chip_gw = max(chip_gws)
            weeks_until_chip = max_chip_gw - current_gw
            horizon_weeks = min(weeks_until_chip + 1, 5)

        # Run optimizer to get recommended transfers
        config = OptimizationConfig(
            horizon_weeks=horizon_weeks,
            free_transfers=free_transfers,
            max_transfers=num_transfers,  # Use provided num_transfers if specified
            available_chips=available_chips
        )
        optimizer = FPLOptimizer(optimization_players, fixtures, teams)
        result = optimizer.optimize_with_transfers(
            current_player_ids,
            config=config
        )

        # Get the optimized squad after transfers
        new_squad = result.current_squad

        # Extract transfers
        first_plan = result.transfer_plans[0] if result.transfer_plans else None
        players_out_ids = set(first_plan.transfers_out) if first_plan else set()
        players_in_ids = set(first_plan.transfers_in) if first_plan else set()

        # Calculate projected squad for next GW
        projected_squad_ids = list(set(current_player_ids) - players_out_ids | players_in_ids)
        projected_squad_players = [p for p in all_players if p.id in projected_squad_ids]

        # Get team stats
        bank = my_team['entry_history']['bank'] / 10
        team_value = my_team['entry_history']['value'] / 10

        # Generate chip recommendations (same as my-team)
        is_first_half = current_gw <= 19
        chip_recommendations = []

        # Wildcard
        if 'Wildcard' in [c.name for c in available_chips]:
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

        # Free Hit
        if 'Free Hit' in [c.name for c in available_chips]:
            if is_first_half:
                chip_recommendations.append(ChipRecommendation(
                    chip_name="Free Hit",
                    recommended_gameweek=18,
                    reasoning="Use on a blank or difficult gameweek (resets at GW20)",
                    priority=3
                ))
            else:
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

        # Bench Boost
        if 'Bench Boost' in [c.name for c in available_chips]:
            if is_first_half:
                chip_recommendations.append(ChipRecommendation(
                    chip_name="Bench Boost",
                    recommended_gameweek=17,
                    reasoning="Use when your full squad has favorable fixtures (resets at GW20)",
                    priority=3
                ))
            else:
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

        # Triple Captain
        if 'Triple Captain' in [c.name for c in available_chips]:
            if is_first_half:
                chip_recommendations.append(ChipRecommendation(
                    chip_name="Triple Captain",
                    recommended_gameweek=18,
                    reasoning="Use when your premium player has an excellent fixture (resets at GW20)",
                    priority=3
                ))
            else:
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

        chip_recommendations.sort(key=lambda x: x.priority)

        # Convert chip enums to strings
        chips_available_strs = []
        if Chip.WILDCARD in available_chips:
            chips_available_strs.append('Wildcard')
        if Chip.FREE_HIT in available_chips:
            chips_available_strs.append('Free Hit')
        if Chip.BENCH_BOOST in available_chips:
            chips_available_strs.append('Bench Boost')
        if Chip.TRIPLE_CAPTAIN in available_chips:
            chips_available_strs.append('Triple Captain')

        return MyTeamResponse(
            team_name=my_info['name'],
            team_value=team_value,
            bank=bank,
            free_transfers=free_transfers,
            current_gameweek=current_gw + 1,  # Show NEXT gameweek
            chips_available=chips_available_strs,
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
                for p in projected_squad_players
            ],
            starting_11_ids=new_squad.starting_11_ids,
            captain_id=new_squad.captain_id,
            vice_captain_id=new_squad.vice_captain_id
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
        my_history = fpl_client.get_my_history()
        bootstrap = fpl_client.get_bootstrap_static()
        current_gw = fpl_client.get_current_gameweek()

        # Check if Free Hit was used in the current gameweek (same logic as my-team endpoint)
        chips_used = my_history.get('chips', [])
        free_hit_gw = None
        for chip in chips_used:
            if chip['name'] == 'freehit':
                free_hit_gw = chip['event']

        # Check if current GW is finished
        events = bootstrap.get('events', [])
        current_gw_data = next((e for e in events if e['id'] == current_gw), None)
        is_gw_finished = current_gw_data and current_gw_data.get('finished', False)

        # Fetch correct team data (reverted team if Free Hit was used in finished GW)
        if free_hit_gw == current_gw and is_gw_finished:
            my_team = fpl_client.get_my_team(gameweek=current_gw - 1)
        else:
            my_team = fpl_client.get_my_team()

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

        # CRITICAL: For transfers, include current squad players even if injured
        # (they're already in your team), but exclude unavailable players for new transfers
        current_squad_players = [p for p in all_players if p.id in current_player_ids]
        available_for_transfer = [p for p in all_players if p.is_available]

        # Combine: current squad + available players (remove duplicates)
        all_player_ids = set([p.id for p in current_squad_players] + [p.id for p in available_for_transfer])
        optimization_players = [p for p in all_players if p.id in all_player_ids]

        # Get fixtures and teams for xP calculation
        current_gw = fpl_client.get_current_gameweek()
        fixtures_data = fpl_client.get_fixtures()
        teams_data = [Team(**t) for t in bootstrap['teams']]
        teams = {t.id: t for t in teams_data}
        fixtures = [Fixture(**f) for f in fixtures_data]

        # Get available chips and convert to Chip enum
        chips_used = my_history.get('chips', [])
        available_chips = []
        chip_name_map = {
            'wildcard': Chip.WILDCARD,
            'freehit': Chip.FREE_HIT,
            'bboost': Chip.BENCH_BOOST,
            '3xc': Chip.TRIPLE_CAPTAIN
        }

        if not any(c['name'] == 'wildcard' for c in chips_used):
            available_chips.append(Chip.WILDCARD)
        if not any(c['name'] == 'freehit' for c in chips_used):
            available_chips.append(Chip.FREE_HIT)
        if not any(c['name'] == 'bboost' for c in chips_used):
            available_chips.append(Chip.BENCH_BOOST)
        if not any(c['name'] == '3xc' for c in chips_used):
            available_chips.append(Chip.TRIPLE_CAPTAIN)

        # Determine planning horizon based on chip timing
        # If Bench Boost or Triple Captain coming soon, extend horizon to include that GW
        horizon_weeks = 2  # Default
        chip_gws = []

        if Chip.BENCH_BOOST in available_chips:
            # Bench Boost recommended in GW17 (first half) or GW36 (second half)
            bb_gw = 17 if current_gw <= 19 else 36
            if bb_gw > current_gw:
                chip_gws.append(bb_gw)

        if Chip.TRIPLE_CAPTAIN in available_chips:
            # Triple Captain for best fixtures or DGWs
            tc_gw = 18 if current_gw <= 19 else 36
            if tc_gw > current_gw:
                chip_gws.append(tc_gw)

        # Extend horizon to include chip gameweeks (up to 5 weeks max)
        if chip_gws:
            max_chip_gw = max(chip_gws)
            weeks_until_chip = max_chip_gw - current_gw
            horizon_weeks = min(weeks_until_chip + 1, 5)  # Plan through chip GW, max 5 weeks

        # Run new optimizer with chip-aware planning
        # Use optimization_players (current squad + available for transfer)
        config = OptimizationConfig(
            horizon_weeks=horizon_weeks,  # Extended for chip preparation
            free_transfers=free_transfers,
            max_transfers=num_transfers,  # Force specific number of transfers if provided
            available_chips=available_chips  # Pass available chips to optimizer
        )
        optimizer = FPLOptimizer(optimization_players, fixtures, teams)
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

        # HORIZON CALCULATIONS: Calculate total gain over planning horizon
        # The optimizer already calculated this - use result.total_expected_points
        # We need to calculate current squad's expected over the same horizon
        current_squad_horizon_xp = 0
        for week_offset in range(horizon_weeks):
            gw = current_gw + week_offset
            week_xp = sum(optimizer._get_xp(p.id, gw) for p in current_players[:11])
            # Add captain bonus
            week_captain_xp = max([optimizer._get_xp(p.id, gw) for p in current_players[:11]], default=0)
            week_xp += week_captain_xp
            # Apply decay for future weeks (same as optimizer)
            decay = config.decay_rate ** week_offset
            current_squad_horizon_xp += week_xp * decay

        # Horizon gain includes the full planning period
        horizon_expected_gain = result.total_expected_points - current_squad_horizon_xp + transfer_cost  # Add back transfer cost since optimizer subtracted it
        horizon_net_gain = horizon_expected_gain - transfer_cost

        # Generate reasoning based on chip planning and transfer value
        reasoning_parts = []

        # Check if planning for chip
        planning_for_chip = False
        if horizon_net_gain > net_gain + 5 and chip_gws:  # Significantly better horizon gain
            chip_gw = max(chip_gws)
            if Chip.BENCH_BOOST in available_chips and chip_gw == (17 if current_gw <= 19 else 36):
                reasoning_parts.append(f"Sets up squad for GW{chip_gw} Bench Boost")
                planning_for_chip = True
            if Chip.TRIPLE_CAPTAIN in available_chips and chip_gw == (18 if current_gw <= 19 else 36):
                reasoning_parts.append(f"Prepares for GW{chip_gw} Triple Captain")
                planning_for_chip = True

        # Explain hit context
        if actual_transfers > free_transfers:
            hits = actual_transfers - free_transfers
            # With escalating penalty, calculate actual cost
            total_hit_cost = 0
            for hit_num in range(1, hits + 1):
                hit_cost = 4 * (1.5 ** (hit_num - 1))
                total_hit_cost += hit_cost

            if planning_for_chip and horizon_net_gain > transfer_cost:
                reasoning_parts.append(f"{hits} hit{'s' if hits > 1 else ''} (-{int(total_hit_cost)}pts) justified for chip prep")
            elif net_gain > transfer_cost:
                reasoning_parts.append(f"{hits} hit{'s' if hits > 1 else ''} (-{int(total_hit_cost)}pts) worth it for +{net_gain:.1f}pts gain")
            else:
                reasoning_parts.append(f"{hits} hit{'s' if hits > 1 else ''} (-{int(total_hit_cost)}pts) NOT recommended")

        if not reasoning_parts:
            if net_gain > 5:
                reasoning_parts.append("Strong immediate gain")
            elif net_gain > 0:
                reasoning_parts.append("Modest improvement")
            else:
                reasoning_parts.append("Hold transfers for next week")

        reasoning = "; ".join(reasoning_parts)

        # Determine recommendation based on HORIZON gain (not just next GW)
        # This accounts for chip preparation value
        if horizon_net_gain > 5:
            recommendation = "make_transfers"
        elif horizon_net_gain > 0 and net_gain > 0:
            recommendation = "make_transfers"
        elif horizon_net_gain >= 0:
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

        # Build transfer suggestions with player stats
        transfers = []
        for out_id, in_id in zip(sorted(players_out_ids), sorted(players_in_ids)):
            player_out = players_dict[out_id]
            player_in = players_dict[in_id]

            # Get fixtures for both players
            player_out_fixtures = get_player_fixtures(player_out.id, player_out.team_id)
            player_in_fixtures = get_player_fixtures(player_in.id, player_in.team_id)

            # Calculate expected points for both players
            player_out_xp = optimizer._get_xp(player_out.id, next_gw)
            player_in_xp = optimizer._get_xp(player_in.id, next_gw)
            points_gain = player_in_xp - player_out_xp

            transfers.append(
                TransferSuggestion(
                    player_out_id=out_id,
                    player_out_name=player_out.name,
                    player_out_position=player_out.position_name,
                    player_out_team=teams[player_out.team_id].short_name,
                    player_out_cost=player_out.cost_millions,
                    player_out_total_points=player_out.total_points,
                    player_out_form=player_out.form,
                    player_out_expected_points=player_out_xp,
                    player_out_fixtures=player_out_fixtures,
                    player_in_id=in_id,
                    player_in_name=player_in.name,
                    player_in_position=player_in.position_name,
                    player_in_team=teams[player_in.team_id].short_name,
                    player_in_cost=player_in.cost_millions,
                    player_in_total_points=player_in.total_points,
                    player_in_form=player_in.form,
                    player_in_expected_points=player_in_xp,
                    player_in_fixtures=player_in_fixtures,
                    cost_change=player_in.cost_millions - player_out.cost_millions,
                    points_gain=points_gain
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
        # CRITICAL: Only consider players in the starting 11 (not bench)
        starting_11_players = [p for p in final_squad_players if p.id in new_squad.starting_11_ids]
        captain_scores = []
        for player in starting_11_players:
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

        # Calculate optimal bench order (positions 12-15)
        # Bench should be ordered by expected points to maximize autosub value
        bench_players = [p for p in final_squad_players if p.id not in new_squad.starting_11_ids]

        # Calculate xP for each bench player
        bench_with_xp = []
        for player in bench_players:
            xp = optimizer._get_xp(player.id, next_gw)
            bench_with_xp.append((player, xp))

        # Sort bench by xP (highest first) - GK must be last per FPL rules
        goalkeepers = [(p, xp) for p, xp in bench_with_xp if p.position_name == 'GKP']
        outfield = [(p, xp) for p, xp in bench_with_xp if p.position_name != 'GKP']

        # Sort outfield by xP (highest first)
        outfield.sort(key=lambda x: x[1], reverse=True)

        # Bench order: outfield players by xP, then GK
        bench_order = [p.id for p, _ in outfield] + [p.id for p, _ in goalkeepers]

        return TransferRecommendationResponse(
            free_transfers=free_transfers,
            num_transfers=actual_transfers,
            transfer_cost=transfer_cost,
            expected_gain=expected_gain,
            net_gain=net_gain,
            horizon_expected_gain=horizon_expected_gain,
            horizon_net_gain=horizon_net_gain,
            planning_horizon_weeks=horizon_weeks,
            recommendation=recommendation,
            reasoning=reasoning,
            transfers=transfers,
            captain_pick=captain_pick,
            vice_captain_pick=vice_captain_pick,
            bench_order=bench_order
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
            # Use captain_mode=True to prioritize proven quality over form
            expected_pts = predictor.calculate_expected_points(player, next_gw, num_gameweeks=1, captain_mode=True)
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


@app.get("/api/free-hit-team", response_model=MyTeamResponse)
async def get_free_hit_team(gameweek: Optional[int] = None):
    """
    Get optimal Free Hit team for next gameweek.

    Free Hit allows unlimited transfers for one gameweek, so we build the best possible
    team from scratch ignoring current squad.
    """
    try:
        fpl_client = get_client()

        # Fetch data
        my_info = fpl_client.get_my_info()
        my_history = fpl_client.get_my_history()
        bootstrap = fpl_client.get_bootstrap_static()
        fixtures_data = fpl_client.get_fixtures()

        # Get all available players
        all_players = [Player(**p) for p in bootstrap['elements']]
        available_players = [p for p in all_players if p.is_available]

        # For Free Hit, filter out bench warmers and non-attacking assets
        # Use ownership as a data-driven proxy for attacking threat
        filtered_players = []
        for p in available_players:
            if p.position_name == 'MID':
                # Midfielders: require minimum ownership threshold (10%) to ensure attacking midfielders
                # Premium players (>=£8.5m) can have lower ownership for differentials
                # This filters out defensive mids and bench fodder
                if p.selected_by_percent >= 10.0 or p.cost_millions >= 8.5:
                    filtered_players.append(p)
            else:
                # Other positions: basic bench warmer filter
                if p.cost_millions >= 4.5 or p.total_points >= 40:
                    filtered_players.append(p)

        available_players = filtered_players

        # Get teams and fixtures
        teams_data = [Team(**t) for t in bootstrap['teams']]
        teams = {t.id: t for t in teams_data}
        fixtures = [Fixture(**f) for f in fixtures_data]

        # Get current gameweek
        current_gw = fpl_client.get_current_gameweek()

        # Free Hit is for NEXT gameweek
        target_gw = current_gw + 1

        # Get team stats
        my_team = fpl_client.get_my_team()
        bank = my_team['entry_history']['bank'] / 10
        team_value = my_team['entry_history']['value'] / 10

        # Get chips
        chips_used = my_history.get('chips', [])
        chips_available = []
        if not any(c['name'] == 'freehit' for c in chips_used):
            chips_available.append('Free Hit')

        # Build optimal team from scratch for NEXT gameweek (Free Hit ignores current team)
        constraints = SquadConstraints(total_budget=100.0)
        optimizer = FPLOptimizer(available_players, fixtures, teams, constraints)

        # Use optimize_squad() with target_gw parameter (now has opposing defender penalty)
        config = OptimizationConfig()
        squad = optimizer.optimize_squad(config=config, target_gw=target_gw)

        # Generate chip recommendations
        chip_recommendations = []
        chip_recommendations.append(ChipRecommendation(
            chip_name="Free Hit",
            recommended_gameweek=target_gw,
            reasoning="Free Hit active - this is your optimal 15-man squad for this gameweek only. Your normal team returns next week.",
            priority=1
        ))

        return MyTeamResponse(
            team_name=my_info.get('name', 'My Team'),
            team_value=team_value,
            bank=bank,
            free_transfers=0,  # Free Hit = unlimited transfers but doesn't affect FT count
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
                    cost_millions=p.cost / 10,
                    total_points=p.total_points,
                    form=p.form,
                    selected_by_percent=p.selected_by_percent,
                    is_available=p.is_available,
                    expected_points=round(optimizer._get_xp(p.id, target_gw), 2)
                )
                for p in squad.players
            ],
            starting_11_ids=squad.starting_11_ids,
            captain_id=squad.captain_id,
            vice_captain_id=squad.vice_captain_id
        )

    except Exception as e:
        logger.error(f"Error in get_free_hit_team: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class GameweekPerformance(BaseModel):
    gameweek: int
    minutes: int
    total_points: int
    goals_scored: int
    assists: int
    clean_sheets: int
    bonus: int


class TeamPlayerStats(BaseModel):
    player_id: int
    player_name: str
    team_name: str
    position: str
    cost: float
    total_points: int
    form: float
    minutes_total: int
    recent_performance: List[GameweekPerformance]


class TeamResponse(BaseModel):
    id: int
    name: str
    short_name: str


@app.get("/api/teams", response_model=List[TeamResponse])
async def get_teams():
    """Get all FPL teams."""
    try:
        fpl_client = get_client()
        bootstrap = fpl_client.get_bootstrap_static()

        teams_data = [Team(**t) for t in bootstrap['teams']]
        return [
            TeamResponse(
                id=t.id,
                name=t.name,
                short_name=t.short_name
            )
            for t in teams_data
        ]
    except Exception as e:
        logger.error(f"Error in get_teams: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/team-analysis", response_model=List[TeamPlayerStats])
async def get_team_analysis(team_id: Optional[int] = None):
    """
    Get detailed performance stats for all players in a specific team,
    including minutes played and recent gameweek performance.
    """
    try:
        fpl_client = get_client()
        bootstrap = fpl_client.get_bootstrap_static()

        # Get teams
        teams_data = [Team(**t) for t in bootstrap['teams']]
        teams_dict = {t.id: t for t in teams_data}

        # Get all players
        all_players = [Player(**p) for p in bootstrap['elements']]

        # Filter by team if specified
        if team_id:
            players_to_analyze = [p for p in all_players if p.team_id == team_id]
        else:
            # If no team specified, return empty list
            players_to_analyze = []

        # Helper function to fetch single player stats
        def fetch_player_stats(player: Player) -> Optional[TeamPlayerStats]:
            try:
                # Fetch player summary (includes history)
                summary = fpl_client.get_element_summary(player.id)
                history = summary.get('history', [])

                # Get last 5 gameweeks
                recent_gws = history[-5:] if len(history) > 0 else []

                # Build recent performance
                recent_performance = []
                for gw_data in recent_gws:
                    recent_performance.append(GameweekPerformance(
                        gameweek=gw_data.get('round', 0),
                        minutes=gw_data.get('minutes', 0),
                        total_points=gw_data.get('total_points', 0),
                        goals_scored=gw_data.get('goals_scored', 0),
                        assists=gw_data.get('assists', 0),
                        clean_sheets=gw_data.get('clean_sheets', 0),
                        bonus=gw_data.get('bonus', 0)
                    ))

                # Calculate total minutes
                minutes_total = sum(gw.get('minutes', 0) for gw in history)

                team = teams_dict.get(player.team_id)
                team_name = team.short_name if team else "Unknown"

                return TeamPlayerStats(
                    player_id=player.id,
                    player_name=player.name,
                    team_name=team_name,
                    position=player.position_name,
                    cost=player.cost_millions,
                    total_points=player.total_points,
                    form=player.form,
                    minutes_total=minutes_total,
                    recent_performance=recent_performance
                )

            except Exception as e:
                logger.warning(f"Failed to fetch stats for player {player.id}: {e}")
                return None

        # Fetch all player stats in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            future_to_player = {executor.submit(fetch_player_stats, player): player for player in players_to_analyze}

            # Collect results as they complete
            result = []
            for future in future_to_player:
                player_stats = future.result()
                if player_stats:
                    result.append(player_stats)

        # Sort by total points descending
        result.sort(key=lambda x: x.total_points, reverse=True)

        return result

    except Exception as e:
        logger.error(f"Error in get_team_analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)