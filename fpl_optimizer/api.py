"""Module for fetching data from the FPL API."""

import requests
import pandas as pd
from typing import Dict, List, Optional


class FPLDataFetcher:
    """Fetches and processes data from the Fantasy Premier League API."""

    BASE_URL = "https://fantasy.premierleague.com/api"

    def __init__(self):
        self.session = requests.Session()
        self.bootstrap_data = None

    def fetch_bootstrap_static(self) -> Dict:
        """
        Fetch the bootstrap-static endpoint containing all basic FPL data.

        Returns:
            Dict containing players, teams, game settings, and more.
        """
        url = f"{self.BASE_URL}/bootstrap-static/"
        response = self.session.get(url)
        response.raise_for_status()
        self.bootstrap_data = response.json()
        return self.bootstrap_data

    def get_players_df(self) -> pd.DataFrame:
        """
        Get all players as a pandas DataFrame.

        Returns:
            DataFrame with player information including stats and pricing.
        """
        if self.bootstrap_data is None:
            self.fetch_bootstrap_static()

        players = pd.DataFrame(self.bootstrap_data['elements'])
        return players

    def get_teams_df(self) -> pd.DataFrame:
        """
        Get all teams as a pandas DataFrame.

        Returns:
            DataFrame with team information.
        """
        if self.bootstrap_data is None:
            self.fetch_bootstrap_static()

        teams = pd.DataFrame(self.bootstrap_data['teams'])
        return teams

    def get_player_fixtures(self, player_id: int) -> List[Dict]:
        """
        Get fixtures for a specific player.

        Args:
            player_id: The FPL player ID.

        Returns:
            List of fixture dictionaries for the player.
        """
        url = f"{self.BASE_URL}/element-summary/{player_id}/"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_fixtures_df(self) -> pd.DataFrame:
        """
        Get all fixtures as a pandas DataFrame.

        Returns:
            DataFrame with fixture information.
        """
        url = f"{self.BASE_URL}/fixtures/"
        response = self.session.get(url)
        response.raise_for_status()
        fixtures = pd.DataFrame(response.json())
        return fixtures

    def get_gameweek_live_data(self, gameweek: int) -> Dict:
        """
        Get live data for a specific gameweek.

        Args:
            gameweek: The gameweek number.

        Returns:
            Dict containing live player data for the gameweek.
        """
        url = f"{self.BASE_URL}/event/{gameweek}/live/"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()


def main():
    """Main function for fetching and displaying FPL data."""
    fetcher = FPLDataFetcher()

    print("Fetching FPL data...")
    fetcher.fetch_bootstrap_static()

    players_df = fetcher.get_players_df()
    print(f"\nTotal players: {len(players_df)}")
    print("\nSample player data:")
    print(players_df[['web_name', 'team', 'now_cost', 'total_points']].head(10))

    teams_df = fetcher.get_teams_df()
    print(f"\nTotal teams: {len(teams_df)}")
    print("\nTeams:")
    print(teams_df[['name', 'strength']].head())


if __name__ == "__main__":
    main()