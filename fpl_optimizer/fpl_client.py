"""FPL API client with caching, rate limiting, and authentication."""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv


class FPLClient:
    """FPL API client with caching and rate limiting."""

    BASE_URL = "https://fantasy.premierleague.com/api"
    CACHE_DIR = Path(__file__).parent / "data" / "cache"
    CACHE_DURATION = timedelta(hours=1)  # Cache expires after 1 hour

    def __init__(self):
        """Initialize the FPL client."""
        load_dotenv()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FPL-Optimizer/1.0'
        })

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests

        # Authentication state
        self.authenticated = False

        # Load cookies if available
        self._load_cookies_from_env()

        # Create cache directory
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _load_cookies_from_env(self):
        """Load authentication cookies from environment variables."""
        # Try OAuth tokens first (new auth system)
        access_token = os.getenv('FPL_ACCESS_TOKEN')
        refresh_token = os.getenv('FPL_REFRESH_TOKEN')

        if access_token:
            # Set OAuth tokens as cookies
            self.session.cookies.set('access_token', access_token, domain='.premierleague.com')
            if refresh_token:
                self.session.cookies.set('refresh_token', refresh_token, domain='.premierleague.com')
            self.authenticated = True
            print("✓ Loaded OAuth tokens from .env")
            return

        # Fall back to old cookie system
        pl_profile = os.getenv('FPL_COOKIE_PL_PROFILE')
        sessionid = os.getenv('FPL_COOKIE_SESSIONID')

        if pl_profile or sessionid:
            if pl_profile:
                self.session.cookies.set('pl_profile', pl_profile, domain='.premierleague.com')
            if sessionid:
                self.session.cookies.set('sessionid', sessionid, domain='fantasy.premierleague.com')
                self.session.cookies.set('sessionid', sessionid, domain='users.premierleague.com')
            self.authenticated = True
            print("✓ Loaded authentication cookies from .env")

    def _wait_for_rate_limit(self):
        """Ensure minimum time between requests."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last_request)

        self.last_request_time = time.time()

    def _make_request(
        self,
        endpoint: str,
        use_cache: bool = True,
        max_retries: int = 3,
        requires_auth: bool = False
    ) -> Dict:
        """
        Make HTTP request with exponential backoff and caching.

        Args:
            endpoint: API endpoint path
            use_cache: Whether to use cached data
            max_retries: Maximum number of retry attempts
            requires_auth: Whether this endpoint requires authentication

        Returns:
            JSON response as dictionary
        """
        # Check authentication if required
        if requires_auth and not self.authenticated:
            self.authenticate()

        # Check cache first
        if use_cache:
            cached_data = self._get_from_cache(endpoint)
            if cached_data:
                return cached_data

        # Make request with exponential backoff
        url = f"{self.BASE_URL}/{endpoint}"

        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                response = self.session.get(url, timeout=10)
                response.raise_for_status()

                data = response.json()

                # Cache successful response
                if use_cache:
                    self._save_to_cache(endpoint, data)

                return data

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to fetch {endpoint} after {max_retries} attempts: {e}")

                # Exponential backoff: 2^attempt seconds
                wait_time = 2 ** attempt
                print(f"Request failed, retrying in {wait_time}s... ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)

    def _get_cache_path(self, endpoint: str) -> Path:
        """Get cache file path for endpoint."""
        # Replace slashes with underscores for filename
        cache_name = endpoint.replace("/", "_").replace("?", "_") + ".json"
        return self.CACHE_DIR / cache_name

    def _get_from_cache(self, endpoint: str) -> Optional[Dict]:
        """Retrieve data from cache if valid."""
        cache_path = self._get_cache_path(endpoint)

        if not cache_path.exists():
            return None

        # Check if cache is still valid
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age > self.CACHE_DURATION:
            return None

        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def _save_to_cache(self, endpoint: str, data: Dict):
        """Save data to cache."""
        cache_path = self._get_cache_path(endpoint)

        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except IOError as e:
            print(f"Warning: Failed to cache data: {e}")

    def authenticate(self) -> bool:
        """
        Authenticate with FPL API using credentials from .env file.

        Returns:
            True if authentication successful, False otherwise
        """
        email = os.getenv('FPL_EMAIL')
        password = os.getenv('FPL_PASSWORD')

        if not email or not password:
            print("Warning: FPL_EMAIL and FPL_PASSWORD not set in .env file")
            return False

        try:
            login_url = "https://users.premierleague.com/accounts/login/"

            # Get CSRF token
            self.session.get(login_url)

            # Login
            payload = {
                'login': email,
                'password': password,
                'app': 'plfpl-web',
                'redirect_uri': 'https://fantasy.premierleague.com/'
            }

            response = self.session.post(login_url, data=payload)

            if response.status_code == 200:
                self.authenticated = True
                print("Successfully authenticated with FPL API")
                return True
            else:
                print(f"Authentication failed with status {response.status_code}")
                return False

        except Exception as e:
            print(f"Authentication error: {e}")
            return False

    def get_bootstrap_static(self, use_cache: bool = True) -> Dict:
        """
        Fetch bootstrap-static data containing players, teams, gameweeks.

        Args:
            use_cache: Whether to use cached data

        Returns:
            Bootstrap static data
        """
        return self._make_request("bootstrap-static/", use_cache=use_cache)

    def get_fixtures(self, use_cache: bool = True) -> List[Dict]:
        """
        Fetch all fixtures.

        Args:
            use_cache: Whether to use cached data

        Returns:
            List of fixtures
        """
        return self._make_request("fixtures/", use_cache=use_cache)

    def get_element_summary(self, player_id: int, use_cache: bool = True) -> Dict:
        """
        Fetch detailed data for a specific player.

        Args:
            player_id: FPL player ID
            use_cache: Whether to use cached data

        Returns:
            Player summary including history and fixtures
        """
        return self._make_request(f"element-summary/{player_id}/", use_cache=use_cache)

    def get_current_gameweek(self) -> int:
        """
        Get the current gameweek number.

        Returns:
            Current gameweek ID
        """
        data = self.get_bootstrap_static()
        events = data.get('events', [])

        for event in events:
            if event.get('is_current'):
                return event['id']

        # If no current event, return next event
        for event in events:
            if event.get('is_next'):
                return event['id']

        return 1  # Fallback

    def get_my_team(self, gameweek: Optional[int] = None) -> Dict:
        """
        Fetch current team for authenticated user.

        Args:
            gameweek: Gameweek number (defaults to current gameweek)

        Returns:
            Team data including picks and transfers
        """
        team_id = os.getenv('FPL_TEAM_ID')

        if not team_id:
            raise ValueError("FPL_TEAM_ID not set in .env file")

        if gameweek is None:
            gameweek = self.get_current_gameweek()

        endpoint = f"entry/{team_id}/event/{gameweek}/picks/"
        return self._make_request(endpoint, use_cache=False, requires_auth=True)

    def get_my_transfers(self) -> Dict:
        """
        Fetch transfer history for authenticated user.

        Returns:
            Transfer history
        """
        team_id = os.getenv('FPL_TEAM_ID')

        if not team_id:
            raise ValueError("FPL_TEAM_ID not set in .env file")

        endpoint = f"entry/{team_id}/transfers/"
        return self._make_request(endpoint, use_cache=False, requires_auth=True)

    def get_my_info(self) -> Dict:
        """
        Fetch team info including chips, free transfers, and bank.

        Returns:
            Team information including:
            - name, summary_overall_points, summary_overall_rank
            - current_event (gameweek)
            - chips: list of chips with status (available/played)
        """
        team_id = os.getenv('FPL_TEAM_ID')

        if not team_id:
            raise ValueError("FPL_TEAM_ID not set in .env file")

        endpoint = f"entry/{team_id}/"
        return self._make_request(endpoint, use_cache=False, requires_auth=True)

    def get_my_history(self) -> Dict:
        """
        Fetch season history including transfers made and points per gameweek.

        Returns:
            Dict containing:
            - current: list of gameweek history this season
            - chips: list of chips used with event number
        """
        team_id = os.getenv('FPL_TEAM_ID')

        if not team_id:
            raise ValueError("FPL_TEAM_ID not set in .env file")

        endpoint = f"entry/{team_id}/history/"
        return self._make_request(endpoint, use_cache=False, requires_auth=True)

    def get_gameweek_live(self, gameweek: int) -> Dict:
        """
        Get live data for a specific gameweek.

        Args:
            gameweek: Gameweek number

        Returns:
            Live gameweek data
        """
        return self._make_request(f"event/{gameweek}/live/", use_cache=False)

    def clear_cache(self):
        """Clear all cached data."""
        for cache_file in self.CACHE_DIR.glob("*.json"):
            cache_file.unlink()
        print("Cache cleared")


def main():
    """Example usage of FPL client."""
    client = FPLClient()

    print("Fetching bootstrap-static data...")
    data = client.get_bootstrap_static()

    print(f"\nTotal players: {len(data['elements'])}")
    print(f"Total teams: {len(data['teams'])}")
    print(f"Current gameweek: {client.get_current_gameweek()}")

    print("\nSample players:")
    for player in data['elements'][:5]:
        print(f"  {player['web_name']} - {player['team']} - £{player['now_cost']/10}M")

    # Try to fetch current team if authenticated
    try:
        print("\nFetching your current team...")
        team = client.get_my_team()
        print(f"Team value: £{team['entry_history']['value']/10}M")
        print(f"Total points: {team['entry_history']['total_points']}")
        print(f"Squad size: {len(team['picks'])}")
    except Exception as e:
        print(f"Could not fetch team (authentication may be needed): {e}")


if __name__ == "__main__":
    main()