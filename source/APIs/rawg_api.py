from dotenv import load_dotenv
import os
import requests
from APIs.api_types import RAWGType
from urllib.parse import quote_plus


class RAWG:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("RAWG_API_KEY")

    def search(self, game_name: str, max_n: int = 1) -> list[RAWGType]:
        """
        Searches for games by name and retrieves details.
        Returns a list of RAWGType objects.
        """
        search_url = f"https://api.rawg.io/api/games?search={quote_plus(game_name)}&key={self.api_key}"

        response = requests.get(search_url)
        response.raise_for_status()

        search_results = response.json().get("results", [])

        if not search_results:
            return []

        # Get details for top max_n results
        rawg_results = []
        for game_summary in search_results[:max_n]:
            game_id = game_summary.get("id")
            details_url = f"https://api.rawg.io/api/games/{game_id}?key={self.api_key}"

            details_response = requests.get(details_url)
            details_response.raise_for_status()
            game_data = details_response.json()

            rawg_obj = RAWGType(
                id=game_data.get("id"),
                name=game_data.get("name"),
                release=game_data.get("released"),
                rawg_rating=float(game_data.get("rating")) / 5.0 if game_data.get("rating") else 0.0,
                metacritic_rating=(float(game_data.get("metacritic")) / 100.0 if game_data.get("metacritic") else 0.0),
                main_story=game_data.get("playtime"),
                platforms=[p.get("platform", {}).get("name") for p in game_data.get("platforms", []) if p.get("platform") is not None],
                genres=[g.get("name") for g in game_data.get("genres", [])],
                keywords=[t.get("name") for t in game_data.get("tags", [])],
                esrb_rating=(game_data.get("esrb_rating", {}).get("name") if game_data.get("esrb_rating") else ""),
                developers=[d.get("name") for d in game_data.get("developers", [])],
                publishers=[p.get("name") for p in game_data.get("publishers", [])],
                description=game_data.get("description_raw"),
            )
            rawg_results.append(rawg_obj)

        return rawg_results
