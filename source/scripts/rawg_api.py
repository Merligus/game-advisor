from dotenv import load_dotenv
import os
import requests
from api_types import RAWGType


class RAWG:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("RAWG_API_KEY")

    def search(self, game_name: str, max_n: int = 1) -> list[RAWGType]:
        """
        Searches for games by name and retrieves details.
        Returns a list of RAWGType objects.
        """
        search_url = f"https://api.rawg.io/api/games?search={game_name}&key={self.api_key}"

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
                released=game_data.get("released"),
                rating=game_data.get("rating"),
                metacritic=game_data.get("metacritic"),
                playtime=game_data.get("playtime"),
                platforms=[
                    p.get("platform", {}).get("name")
                    for p in game_data.get("platforms", [])
                ],
                genres=[g.get("name") for g in game_data.get("genres", [])],
                tags=[t.get("name") for t in game_data.get("tags", [])],
                esrb_rating=game_data.get("esrb_rating", {}).get("name"),
                developers=[d.get("name") for d in game_data.get("developers", [])],
                publishers=[p.get("name") for p in game_data.get("publishers", [])],
                description=game_data.get("description_raw"),
            )
            rawg_results.append(rawg_obj)

        return rawg_results


# --- Execution ---
if __name__ == "__main__":
    rawg = RAWG()
    results = rawg.search("Minecraft", max_n=1)

    for game in results:
        print("\n--- Game Found ---")
        print(f"ID: {game.id}")
        print(f"Name: {game.name}")
        print(f"Released: {game.released}")
        print(f"Rating: {game.rating}")
        print(f"Metacritic: {game.metacritic}")
        print(f"Playtime: {game.playtime}")
        print(f"Platforms: {game.platforms}")
        print(f"Genres: {game.genres}")
        print(f"Tags: {game.tags}")
        print(f"ESRB Rating: {game.esrb_rating}")
        print(f"Developers: {game.developers}")
        print(f"Publishers: {game.publishers}")
        print(f"Description: {game.description}")
