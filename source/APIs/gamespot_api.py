import requests
from dotenv import load_dotenv
import os
import xml.etree.ElementTree as ET
from APIs.api_types import GamespotType
from datetime import datetime
from urllib.parse import quote_plus


class Gamespot:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GAMESPOT_API_KEY")

        # Format pattern to get release date 2025-04-24 12:00:00
        self.format_pattern = "%Y-%m-%d %H:%M:%S"

    def search(self, game_name: str, max_n: int = 1) -> list[GamespotType]:
        """
        Searches for games by name and retrieves details.
        Returns a list of GamespotType objects.
        """
        search_url = f"https://www.gamespot.com/api/games/?limit={max_n}&filter=name:{quote_plus(game_name)}&api_key={self.api_key}"

        # Set headers for the request
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()

        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return []

        # Parse XML response
        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as e:
            print(f"XML Parse Error: {e}")
            print(f"Response: {response.text[:500]}")
            return []

        # Get details for top max_n results
        gamespot_results = []

        # Extract reviews from this page
        for game in root.findall(".//game"):
            gamespot_obj = GamespotType(
                id=int(game.findtext("id", 0)),
                name=game.findtext("name"),
                themes=[theme.findtext("name") for theme in game.findall("theme")],
                release=datetime.strptime(game.findtext("release_date"), self.format_pattern).strftime("%Y-%m-%d"),
                genres=[genre.findtext("name") for genre in game.findall("genre")],
                cover_url=[game.findtext("image/original")],
                description=game.findtext("description"),
            )
            gamespot_results.append(gamespot_obj)
        return gamespot_results
