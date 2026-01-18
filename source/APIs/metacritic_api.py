from dotenv import load_dotenv
import os
import requests
from APIs.api_types import MetacriticType
from difflib import SequenceMatcher
from urllib.parse import quote


class Metacritic:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("METACRITIC_API_KEY")

    def _score_ratio(self, score_obj: dict):
        score = score_obj.get("score", 0)
        num = 0 if score is None else score
        max = score_obj.get("max", 1)
        den = 1 if max is None else max
        return float(num) / float(den)

    def search(self, game_name: str, max_n: int = 1) -> list[MetacriticType]:
        """
        Searches for games by name and retrieves details.
        Returns a list of MetacriticType objects.
        """
        # Find name of the game in the Metacritic API
        find_url = f"https://backend.metacritic.com/finder/metacritic/search/{quote(game_name)}/web?apiKey={self.api_key}&limit={max_n+5}&offset=0"
        find_response = requests.get(find_url)
        find_response.raise_for_status()
        find_data = find_response.json()

        # No result found
        if not "data" in find_data:
            return []

        # Start the return array
        metacritic_results = []
        for item in find_data["data"]["items"]:
            if item["type"] == "game-title":
                # Match between names
                name_ratio = SequenceMatcher(None, item["title"], game_name).ratio()

                # Get game metadata
                game_slug = item["slug"]
                game_url = f"https://backend.metacritic.com/composer/metacritic/pages/games/{game_slug}/web?contentOnly=true"
                game_response = requests.get(game_url)
                game_response.raise_for_status()
                game_data = game_response.json()

                if not "components" in game_data or len(game_data["components"]) < 8:
                    continue
                if not "data" in game_data["components"][0]:
                    continue
                if not "data" in game_data["components"][6]:
                    continue
                if not "data" in game_data["components"][8]:
                    continue

                metadata = game_data["components"][0]["data"]["item"]
                critics_score = self._score_ratio(game_data["components"][6]["data"]["item"])
                user_score = self._score_ratio(game_data["components"][8]["data"]["item"])

                metacritic_obj = MetacriticType(
                    name=metadata.get("title"),
                    release=metadata.get("releaseDate", ""),
                    developers=[company["name"] for company in metadata.get("production", {}).get("companies", []) if company["typeName"] == "Developer"],
                    publishers=[company["name"] for company in metadata.get("production", {}).get("companies", []) if company["typeName"] == "Publisher"],
                    genres=[genre["name"] for genre in metadata.get("genres", [])],
                    platforms=[platform["name"] for platform in metadata.get("platforms", [])],
                    metacritic_rating=critics_score,
                    user_rating=user_score,
                )
                metacritic_results.append((metacritic_obj, name_ratio))

        # Sort by ratio of the name
        sorted_data = sorted(metacritic_results, key=lambda x: x[1], reverse=True)

        # Only return the max_n results
        return [obj for obj, ratio in sorted_data][:max_n]
