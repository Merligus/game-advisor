import datetime
import os

from dotenv import load_dotenv

from APIs.api_types import IGDBType
from APIs.http_client import HttpClient


class IGDB:
    def __init__(self):
        load_dotenv()
        self.client_id = os.getenv("IGDB_CLIENT_ID")
        self.client_secret = os.getenv("IGDB_CLIENT_SECRET")
        self.access_token = None
        self.http = HttpClient()

        # Format pattern to get release date 2025-04-24 00:00:00+00:00
        self.format_pattern = "%Y-%m-%d %H:%M:%S+00:00"

    def _get_access_token(self):
        """
        Authenticates with Twitch to get the Bearer Token required for IGDB.
        """
        response = self.http.post(
            "https://id.twitch.tv/oauth2/token",
            params={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "client_credentials",
            },
        )
        self.access_token = response.json()["access_token"]
        return self.access_token

    def _query(self, body: str):
        """POST an APICALYPSE query; on 401 (expired token) refresh once and retry."""
        import requests as _requests

        headers = {
            "Client-ID": self.client_id,
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "text/plain",
        }
        try:
            return self.http.post("https://api.igdb.com/v4/games", headers=headers, data=body).json()
        except _requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 401:
                self._get_access_token()
                headers["Authorization"] = f"Bearer {self.access_token}"
                return self.http.post("https://api.igdb.com/v4/games", headers=headers, data=body).json()
            raise

    def search(self, game_name: str, max_n: int = 1) -> list[IGDBType]:
        """
        Searches for games by name and retrieves details.
        Returns a list of IGDBType objects.
        """
        if not self.access_token:
            self._get_access_token()

        # IGDB's APICALYPSE query language wants the raw search string with only
        # backslashes and double-quotes escaped — NOT URL-encoded. Using quote()
        # here (the old behavior) turned 'The Witcher 3: Wild Hunt' into
        # 'The%20Witcher%203%3A...' which matched nothing; only punctuation-free
        # names happened to work.
        safe_name = game_name.replace("\\", "\\\\").replace('"', '\\"')
        body = f"""
            fields name, game_modes.name, game_type.type, keywords.name, language_supports.language.name, platforms.name, player_perspectives.name, themes.name, rating, summary, first_release_date, genres.name, cover.url, involved_companies.company.name, involved_companies.developer, involved_companies.publisher;
            search "{safe_name}";
            limit {max_n};
        """
        results = self._query(body)

        igdb_results = []
        for game in results:
            companies = game.get("involved_companies", []) or []
            developers = [c["company"]["name"] for c in companies
                          if c.get("developer") and c.get("company", {}).get("name")]
            publishers = [c["company"]["name"] for c in companies
                          if c.get("publisher") and c.get("company", {}).get("name")]
            igdb_obj = IGDBType(
                name=game.get("name"),
                game_modes=[g["name"] for g in game.get("game_modes", [])],
                game_type=game.get("game_type", {}).get("type"),
                keywords=[g["name"] for g in game.get("keywords", [])],
                language_supports=[g["language"]["name"] for g in game.get("language_supports", [])],
                platforms=[g["name"] for g in game.get("platforms", [])],
                player_perspectives=[g["name"] for g in game.get("player_perspectives", [])],
                themes=[g["name"] for g in game.get("themes", [])],
                igdb_rating=float(game.get("rating")) / 100.0 if game.get("rating") else 0.0,
                release=datetime.datetime.fromtimestamp(int(game.get("first_release_date", "0")), datetime.timezone.utc).strftime("%Y-%m-%d"),
                genres=[g["name"] for g in game.get("genres", [])],
                cover_url=[game.get("cover", {}).get("url")],
                description=game.get("summary"),
                developers=developers,
                publishers=publishers,
            )
            igdb_results.append(igdb_obj)

        return igdb_results
