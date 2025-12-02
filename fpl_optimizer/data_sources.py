"""
Data Sources Module - Scrape advanced stats from Understat and FBref.

This module provides scrapers to fetch:
- Understat: xG, xA, shots, shot quality
- FBref: Detailed passing, defensive actions, set pieces
"""

import logging
import re
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup
import json

from .expected_points import PlayerStats, TeamStrength

logger = logging.getLogger(__name__)


# =============================================================================
# Understat Scraper
# =============================================================================

class UnderstatScraper:
    """
    Scrape xG, xA, shots from Understat.

    Understat provides JSON data embedded in script tags.
    """

    BASE_URL = "https://understat.com"

    def __init__(self, season: str = "2024"):
        """
        Initialize Understat scraper.

        Args:
            season: Season year (e.g., "2024" for 2024/25)
        """
        self.season = season
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.rate_limit_delay = 2.0  # Seconds between requests
        self.last_request_time = 0

    def _rate_limit(self):
        """Ensure minimum time between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch page with error handling."""
        try:
            self._rate_limit()
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def _extract_json_from_script(self, html: str, var_name: str) -> Optional[List[Dict]]:
        """
        Extract JSON data from embedded script tags.

        Understat embeds data like: var playersData = JSON.parse('[...]')
        """
        try:
            # Find the script containing the variable
            pattern = rf"var {var_name} = JSON\.parse\('(.+?)'\)"
            match = re.search(pattern, html)

            if not match:
                return None

            # Decode the JSON string (it's encoded)
            json_str = match.group(1)
            json_str = json_str.encode('utf-8').decode('unicode_escape')

            # Parse JSON
            data = json.loads(json_str)
            return data
        except Exception as e:
            logger.error(f"Failed to extract JSON for {var_name}: {e}")
            return None

    def get_league_players(self, league: str = "EPL") -> Dict[str, Dict]:
        """
        Get all players from a league with their stats.

        Args:
            league: League name (EPL, La_Liga, Bundesliga, etc.)

        Returns:
            Dict mapping player_name -> player_stats
        """
        url = f"{self.BASE_URL}/league/{league}/{self.season}"
        html = self._fetch_page(url)

        if not html:
            return {}

        players_data = self._extract_json_from_script(html, 'playersData')

        if not players_data:
            logger.warning("No players data found on Understat")
            return {}

        # Parse into dict
        players = {}
        for player in players_data:
            player_name = player.get('player_name', '')

            # Extract key stats
            players[player_name] = {
                'player_id': player.get('id'),
                'team_title': player.get('team_title'),
                'position': player.get('position'),
                'games': int(player.get('games', 0)),
                'time': int(player.get('time', 0)),  # Total minutes
                'goals': int(player.get('goals', 0)),
                'assists': int(player.get('assists', 0)),
                'xG': float(player.get('xG', 0)),
                'xA': float(player.get('xA', 0)),
                'shots': int(player.get('shots', 0)),
                'key_passes': int(player.get('key_passes', 0)),
                'yellow_cards': int(player.get('yellow_cards', 0)),
                'red_cards': int(player.get('red_cards', 0)),
            }

        logger.info(f"Fetched {len(players)} players from Understat")
        return players

    def get_team_stats(self, league: str = "EPL") -> Dict[str, Dict]:
        """
        Get team-level stats from Understat.

        Args:
            league: League name

        Returns:
            Dict mapping team_name -> team_stats
        """
        url = f"{self.BASE_URL}/league/{league}/{self.season}"
        html = self._fetch_page(url)

        if not html:
            return {}

        teams_data = self._extract_json_from_script(html, 'teamsData')

        if not teams_data:
            logger.warning("No teams data found on Understat")
            return {}

        teams = {}
        for team_id, team in teams_data.items():
            team_name = team.get('title', '')

            # Extract key stats
            teams[team_name] = {
                'team_id': team_id,
                'games': int(team.get('matches', 0)),
                'xG': float(team.get('xG', 0)),
                'xGA': float(team.get('xGA', 0)),
                'scored': int(team.get('scored', 0)),
                'conceded': int(team.get('missed', 0)),
                'wins': int(team.get('wins', 0)),
                'draws': int(team.get('draws', 0)),
                'loses': int(team.get('loses', 0)),
                'pts': int(team.get('pts', 0)),
            }

        logger.info(f"Fetched {len(teams)} teams from Understat")
        return teams


# =============================================================================
# FBref Scraper
# =============================================================================

class FBrefScraper:
    """
    Scrape advanced stats from FBref.

    FBref provides detailed stats in HTML tables.
    Note: FBref has stricter rate limiting - be respectful!
    """

    BASE_URL = "https://fbref.com"

    def __init__(self, season: str = "2024-2025"):
        """
        Initialize FBref scraper.

        Args:
            season: Season string (e.g., "2024-2025")
        """
        self.season = season
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.rate_limit_delay = 3.0  # FBref requires longer delays
        self.last_request_time = 0

    def _rate_limit(self):
        """Ensure minimum time between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch page and parse with BeautifulSoup."""
        try:
            self._rate_limit()
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def get_premier_league_stats(self) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """
        Get Premier League player and team stats from FBref.

        Returns:
            (player_stats_dict, team_stats_dict)
        """
        # Premier League URL (adjust comp ID if needed)
        url = f"{self.BASE_URL}/en/comps/9/stats/Premier-League-Stats"

        soup = self._fetch_page(url)
        if not soup:
            return {}, {}

        players = self._parse_standard_stats_table(soup)
        logger.info(f"Fetched {len(players)} players from FBref")

        # For team stats, would need to navigate to team pages
        # Placeholder for now
        teams = {}

        return players, teams

    def _parse_standard_stats_table(self, soup: BeautifulSoup) -> Dict[str, Dict]:
        """Parse the standard stats table from FBref."""
        players = {}

        # Find the stats table
        table = soup.find('table', {'id': 'stats_standard'})
        if not table:
            logger.warning("Standard stats table not found")
            return players

        tbody = table.find('tbody')
        if not tbody:
            return players

        for row in tbody.find_all('tr'):
            # Skip header rows
            if 'thead' in row.get('class', []):
                continue

            try:
                # Extract player name
                player_cell = row.find('th', {'data-stat': 'player'})
                if not player_cell:
                    continue

                player_name = player_cell.text.strip()

                # Extract stats
                stats = {}
                for cell in row.find_all('td'):
                    stat_name = cell.get('data-stat')
                    stat_value = cell.text.strip()

                    if stat_name:
                        stats[stat_name] = stat_value

                # Parse key stats
                players[player_name] = {
                    'nation': stats.get('nationality', ''),
                    'position': stats.get('position', ''),
                    'team': stats.get('team', ''),
                    'age': stats.get('age', ''),
                    'games': self._parse_int(stats.get('games', '0')),
                    'games_starts': self._parse_int(stats.get('games_starts', '0')),
                    'minutes': self._parse_int(stats.get('minutes', '0')),
                    'goals': self._parse_int(stats.get('goals', '0')),
                    'assists': self._parse_int(stats.get('assists', '0')),
                    'pens_made': self._parse_int(stats.get('pens_made', '0')),
                    'pens_att': self._parse_int(stats.get('pens_att', '0')),
                    'cards_yellow': self._parse_int(stats.get('cards_yellow', '0')),
                    'cards_red': self._parse_int(stats.get('cards_red', '0')),
                }

            except Exception as e:
                logger.warning(f"Failed to parse row: {e}")
                continue

        return players

    @staticmethod
    def _parse_int(value: str) -> int:
        """Parse integer from string, handling empty/invalid values."""
        try:
            return int(value.replace(',', ''))
        except (ValueError, AttributeError):
            return 0

    @staticmethod
    def _parse_float(value: str) -> float:
        """Parse float from string, handling empty/invalid values."""
        try:
            return float(value.replace(',', ''))
        except (ValueError, AttributeError):
            return 0.0


# =============================================================================
# Data Aggregator
# =============================================================================

class DataAggregator:
    """
    Aggregate data from multiple sources into PlayerStats and TeamStrength.

    Combines:
    - FPL API (baseline)
    - Understat (xG, xA)
    - FBref (detailed stats)
    """

    def __init__(
        self,
        fpl_players: List,  # From FPL API
        fpl_teams: Dict[int, any],  # From FPL API
        understat_scraper: Optional[UnderstatScraper] = None,
        fbref_scraper: Optional[FBrefScraper] = None,
    ):
        """
        Initialize aggregator.

        Args:
            fpl_players: List of Player objects from FPL API
            fpl_teams: Dict of team_id -> Team from FPL API
            understat_scraper: Understat scraper instance
            fbref_scraper: FBref scraper instance
        """
        self.fpl_players = {p.id: p for p in fpl_players}
        self.fpl_teams = fpl_teams
        self.understat = understat_scraper
        self.fbref = fbref_scraper

        # Name mapping (FPL web_name -> Understat/FBref name)
        # This is the tricky part - names don't always match
        self._name_mapping = self._build_name_mapping()

    def _build_name_mapping(self) -> Dict[str, str]:
        """
        Build mapping from FPL names to Understat/FBref names.

        This is complex because:
        - FPL uses "web_name" (e.g., "Salah")
        - Understat uses full name (e.g., "Mohamed Salah")
        - FBref uses full name with diacritics

        For now, use fuzzy matching or manual mappings.
        """
        # Placeholder: would implement fuzzy matching here
        # For production, would use a library like fuzzywuzzy or manual CSV mapping
        mapping = {}

        # Example manual mappings (would expand this)
        mapping_overrides = {
            'Salah': 'Mohamed Salah',
            'Haaland': 'Erling Haaland',
            'De Bruyne': 'Kevin De Bruyne',
            'Son': 'Heung-Min Son',
            'Fernandes': 'Bruno Fernandes',
            'Saka': 'Bukayo Saka',
            'Palmer': 'Cole Palmer',
            # Add more as needed
        }

        for fpl_id, player in self.fpl_players.items():
            fpl_name = player.name  # web_name

            # Check manual overrides
            if fpl_name in mapping_overrides:
                mapping[fpl_name] = mapping_overrides[fpl_name]
            else:
                # Default: assume names match (will fail for many)
                mapping[fpl_name] = fpl_name

        return mapping

    def build_player_stats(self) -> Dict[int, PlayerStats]:
        """
        Build PlayerStats for all FPL players by combining data sources.

        Returns:
            Dict mapping player_id -> PlayerStats
        """
        logger.info("Building player stats from multiple sources...")

        # Fetch Understat data
        understat_players = {}
        if self.understat:
            understat_players = self.understat.get_league_players()

        # Fetch FBref data
        fbref_players = {}
        if self.fbref:
            fbref_players, _ = self.fbref.get_premier_league_stats()

        # Build PlayerStats for each FPL player
        player_stats = {}

        for player_id, fpl_player in self.fpl_players.items():
            # Map FPL name to external sources
            fpl_name = fpl_player.name
            external_name = self._name_mapping.get(fpl_name, fpl_name)

            # Get data from each source
            understat_data = understat_players.get(external_name, {})
            fbref_data = fbref_players.get(external_name, {})

            # Calculate per-90 stats
            minutes = understat_data.get('time', 0) or fpl_player.minutes or 1
            minutes_90 = minutes / 90 if minutes > 0 else 1

            # Build PlayerStats
            stats = PlayerStats(
                player_id=player_id,
                position=fpl_player.position_name,
                team_id=fpl_player.team_id,

                # xG/xA from Understat
                xg_per_90=understat_data.get('xG', 0) / minutes_90 if minutes_90 > 0 else 0,
                xa_per_90=understat_data.get('xA', 0) / minutes_90 if minutes_90 > 0 else 0,
                xgi_per_90=(understat_data.get('xG', 0) + understat_data.get('xA', 0)) / minutes_90 if minutes_90 > 0 else 0,

                # Shots from Understat
                shots_per_90=understat_data.get('shots', 0) / minutes_90 if minutes_90 > 0 else 0,
                shots_on_target_per_90=understat_data.get('shots', 0) * 0.35 / minutes_90 if minutes_90 > 0 else 0,  # Estimate
                key_passes_per_90=understat_data.get('key_passes', 0) / minutes_90 if minutes_90 > 0 else 0,

                # Minutes data
                total_minutes=minutes,
                avg_minutes_per_game=minutes / max(1, understat_data.get('games', 1)),
                games_started=fbref_data.get('games_starts', 0) or fpl_player.minutes // 70,  # Estimate

                # Penalties
                is_penalty_taker=self._is_penalty_taker(fpl_player),
                penalty_order=self._get_penalty_order(fpl_player),
                penalties_taken=fbref_data.get('pens_att', 0),
                penalties_scored=fbref_data.get('pens_made', 0),

                # Bonus (from FPL)
                avg_bonus_per_game=fpl_player.total_points / max(1, fpl_player.minutes // 90) if hasattr(fpl_player, 'bonus') else 0,

                # Form (from FPL form field - last 5 games average)
                form_xgi_per_90=fpl_player.form * 0.5 if fpl_player.form > 0 else 0,  # Rough estimate
                form_minutes=fpl_player.minutes / max(1, fpl_player.minutes // 90) if fpl_player.minutes > 0 else 0,
                form_games=5,  # FPL form is last 5 games
            )

            player_stats[player_id] = stats

        logger.info(f"Built stats for {len(player_stats)} players")
        return player_stats

    def build_team_strength(self) -> Dict[int, TeamStrength]:
        """
        Build TeamStrength for all teams by combining data sources.

        Returns:
            Dict mapping team_id -> TeamStrength
        """
        logger.info("Building team strength from multiple sources...")

        # Fetch Understat team data
        understat_teams = {}
        if self.understat:
            understat_teams = self.understat.get_team_stats()

        # Build TeamStrength for each FPL team
        team_strength = {}

        for team_id, fpl_team in self.fpl_teams.items():
            team_name = fpl_team.name

            # Get Understat data (need to map team names)
            understat_data = self._find_understat_team(team_name, understat_teams)

            # Calculate per-90 stats
            games = understat_data.get('games', 1) if understat_data else 1

            if understat_data:
                xg_total = understat_data.get('xG', 0)
                xga_total = understat_data.get('xGA', 0)
                goals_total = understat_data.get('scored', 0)
                conceded_total = understat_data.get('conceded', 0)
            else:
                # Fallback to FPL strength ratings (normalized)
                xg_total = (fpl_team.strength_attack_home + fpl_team.strength_attack_away) / 200 * games
                xga_total = (fpl_team.strength_defence_home + fpl_team.strength_defence_away) / 200 * games
                goals_total = xg_total
                conceded_total = xga_total

            strength = TeamStrength(
                team_id=team_id,
                team_name=team_name,

                # Overall stats
                xg_per_90=(xg_total / games) * 90 / 90 if games > 0 else 0,  # Already per-game, just normalize
                xga_per_90=(xga_total / games) * 90 / 90 if games > 0 else 0,
                goals_per_90=(goals_total / games) * 90 / 90 if games > 0 else 0,
                goals_conceded_per_90=(conceded_total / games) * 90 / 90 if games > 0 else 0,

                # Home/away splits (use FPL strength as proxy if Understat not available)
                home_xg_per_90=fpl_team.strength_attack_home / 1000 * 1.5,  # Normalize to ~1.5 avg
                away_xg_per_90=fpl_team.strength_attack_away / 1000 * 1.3,
                home_xga_per_90=fpl_team.strength_defence_home / 1000 * 1.2,
                away_xga_per_90=fpl_team.strength_defence_away / 1000 * 1.4,

                # Clean sheets (from FPL or estimate)
                clean_sheet_percentage=self._estimate_cs_percentage(fpl_team),

                # Penalties (estimate from league average)
                penalties_won_per_90=0.08,  # EPL average
                penalties_conceded_per_90=0.08,
            )

            team_strength[team_id] = strength

        logger.info(f"Built strength for {len(team_strength)} teams")
        return team_strength

    def _find_understat_team(self, fpl_name: str, understat_teams: Dict) -> Optional[Dict]:
        """Find Understat team data by FPL team name."""
        # Team name mapping
        name_map = {
            'Man City': 'Manchester City',
            'Man Utd': 'Manchester United',
            'Spurs': 'Tottenham',
            'Newcastle': 'Newcastle United',
            'West Ham': 'West Ham',
            'Wolves': 'Wolverhampton Wanderers',
            'Nott\'m Forest': 'Nottingham Forest',
            'Brighton': 'Brighton',
            'Leicester': 'Leicester',
            # Add more as needed
        }

        mapped_name = name_map.get(fpl_name, fpl_name)

        # Try exact match
        if mapped_name in understat_teams:
            return understat_teams[mapped_name]

        # Try fuzzy match (simplified)
        for team_name, data in understat_teams.items():
            if fpl_name.lower() in team_name.lower() or team_name.lower() in fpl_name.lower():
                return data

        return None

    def _estimate_cs_percentage(self, team) -> float:
        """Estimate clean sheet percentage from FPL strength."""
        # Higher defensive strength = more clean sheets
        avg_def_strength = (team.strength_defence_home + team.strength_defence_away) / 2

        # Map strength (1000-1400 range) to CS% (20-45% range)
        cs_pct = 20 + ((avg_def_strength - 1000) / 400) * 25
        return max(15, min(50, cs_pct))

    def _is_penalty_taker(self, player) -> bool:
        """Determine if player is a penalty taker (heuristic)."""
        # Look for high-priced attackers with penalties scored
        if hasattr(player, 'penalties_scored') and player.penalties_scored > 0:
            return True

        # Heuristic: premium players (>£9m) in attacking positions
        if player.cost_millions >= 9.0 and player.position_name in ('MID', 'FWD'):
            return True

        return False

    def _get_penalty_order(self, player) -> int:
        """Get penalty order (1 = first choice, 2 = backup, 0 = not taker)."""
        if not self._is_penalty_taker(player):
            return 0

        # Heuristic: premium players are first choice
        if player.cost_millions >= 10.0:
            return 1
        else:
            return 2  # Backup or shared


# =============================================================================
# Convenience Function
# =============================================================================

def build_advanced_stats(
    fpl_players: List,
    fpl_teams: Dict,
    use_understat: bool = True,
    use_fbref: bool = False,  # FBref is slower, make it optional
) -> Tuple[Dict[int, PlayerStats], Dict[int, TeamStrength]]:
    """
    Convenience function to build advanced stats from all sources.

    Args:
        fpl_players: List of FPL Player objects
        fpl_teams: Dict of FPL Team objects
        use_understat: Whether to scrape Understat
        use_fbref: Whether to scrape FBref (slower)

    Returns:
        (player_stats_dict, team_strength_dict)
    """
    understat = UnderstatScraper() if use_understat else None
    fbref = FBrefScraper() if use_fbref else None

    aggregator = DataAggregator(fpl_players, fpl_teams, understat, fbref)

    player_stats = aggregator.build_player_stats()
    team_strength = aggregator.build_team_strength()

    return player_stats, team_strength