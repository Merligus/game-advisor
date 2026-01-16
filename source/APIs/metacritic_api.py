from dotenv import load_dotenv
import os
import requests
from APIs.api_types import MetacriticType
from difflib import SequenceMatcher


class Metacritic:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("METACRITIC_API_KEY")

    def search(self, game_name: str, max_n: int = 1) -> list[MetacriticType]:
        """
        Searches for games by name and retrieves details.
        Returns a list of MetacriticType objects.
        """
        # Find name of the game in the Metacritic API
        find_url = f"https://backend.metacritic.com/finder/metacritic/search/{game_name}/web?apiKey={self.api_key}&limit={max_n+5}&offset=0"
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

                if not "components" in game_data:
                    continue

                metadata = game_data["components"][0]["data"]["item"]
                critics_score = game_data["components"][6]["data"]["item"]
                user_score = game_data["components"][8]["data"]["item"]

                metacritic_obj = MetacriticType(
                    name=metadata.get("title"),
                    release_date=metadata.get("releaseDate", ""),
                    developers=[
                        company["name"]
                        for company in metadata.get("production", {}).get(
                            "companies", []
                        )
                        if company["typeName"] == "Developer"
                    ],
                    publishers=[
                        company["name"]
                        for company in metadata.get("production", {}).get(
                            "companies", []
                        )
                        if company["typeName"] == "Publisher"
                    ],
                    genres=[genre["name"] for genre in metadata.get("genres", [])],
                    platforms=[
                        platform["name"] for platform in metadata.get("platforms", [])
                    ],
                    critic_score=critics_score.get("score", 0)
                    / critics_score.get("max", 1),
                    user_score=user_score.get("score", 0) / user_score.get("max", 1),
                )
                print(f"ratio: {name_ratio} name: {metacritic_obj.name}")
                metacritic_results.append((metacritic_obj, name_ratio))

        # Sort by ratio of the name
        sorted_data = sorted(metacritic_results, key=lambda x: x[1], reverse=True)

        # Only return the max_n results
        return [obj for obj, ratio in sorted_data][:max_n]
