import os
import requests
from dotenv import load_dotenv
from APIs.api_types import IGDBType
import datetime
from urllib.parse import quote


class IGDB:
    def __init__(self):
        load_dotenv()
        self.client_id = os.getenv("IGDB_CLIENT_ID")
        self.client_secret = os.getenv("IGDB_CLIENT_SECRET")
        self.access_token = None
        
        # Format pattern to get release date 2025-04-24 00:00:00+00:00
        self.format_pattern = "%Y-%m-%d %H:%M:%S+00:00"

    def _get_access_token(self):
        """
        Authenticates with Twitch to get the Bearer Token required for IGDB.
        """
        auth_url = "https://id.twitch.tv/oauth2/token"
        params = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials",
        }

        response = requests.post(auth_url, params=params)

        if response.status_code != 200:
            raise Exception(f"Authentication failed: {response.text}")

        self.access_token = response.json()["access_token"]
        return self.access_token

    def search(self, game_name: str, max_n: int = 1) -> list[IGDBType]:
        """
        Searches for games by name and retrieves details.
        Returns a list of IGDBType objects.
        """
        if not self.access_token:
            self._get_access_token()

        url = "https://api.igdb.com/v4/games"

        headers = {
            "Client-ID": self.client_id,
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "text/plain",
        }

        body = f"""
            fields name, game_modes.name, game_type.type, keywords.name, language_supports.language.name, platforms.name, player_perspectives.name, themes.name, rating, summary, first_release_date, genres.name, cover.url;
            search "{quote(game_name)}";
            limit {max_n};
        """

        response = requests.post(url, headers=headers, data=body)

        if response.status_code != 200:
            raise Exception(f"IGDB Query failed: {response.text}")

        results = response.json()
        igdb_results = []

        for game in results:
            igdb_obj = IGDBType(
                id=game.get("id"),
                name=game.get("name"),
                game_modes=[g["name"] for g in game.get("game_modes", [])],
                game_type=game.get("game_type", {}).get("type"),
                keywords=[g["name"] for g in game.get("keywords", [])],
                language_supports=[g["language"]["name"] for g in game.get("language_supports", [])],
                platforms=[g["name"] for g in game.get("platforms", [])],
                player_perspectives=[g["name"] for g in game.get("player_perspectives", [])],
                themes=[g["name"] for g in game.get("themes", [])],
                igdb_rating=float(game.get("rating")) / 100. if game.get("rating") else 0.0,
                release=datetime.datetime.fromtimestamp(int(game.get("first_release_date", "0")), datetime.timezone.utc).strftime("%Y-%m-%d"),
                genres=[g["name"] for g in game.get("genres", [])],
                cover_url=[game.get("cover", {}).get("url")],
                description=game.get("summary"),
            )
            igdb_results.append(igdb_obj)

        return igdb_results
