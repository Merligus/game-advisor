import os
from difflib import SequenceMatcher
from urllib.parse import quote

from dotenv import load_dotenv

from APIs.api_types import MetacriticType
from APIs.http_client import HttpClient


class Metacritic:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("METACRITIC_API_KEY")
        self.http = HttpClient()

    def _score_ratio(self, score_obj: dict):
        score = score_obj.get("score", 0)
        num = 0 if score is None else score
        max = score_obj.get("max", 1)
        den = 1 if max is None else max
        return float(num) / float(den)

    @staticmethod
    def _find_score_items(components: list) -> tuple[dict | None, dict | None]:
        """Locate the critic- and user-score items by component identity instead
        of the historical hardcoded components[6]/[8] positions (which silently
        broke whenever Metacritic re-ordered the page layout). A score item is a
        dict carrying 'score'/'max'; critic vs user is read from the component's
        name/title metadata. Falls back to the old fixed indices when the scan
        finds nothing.
        """
        critic, user = None, None
        for comp in components:
            item = (comp.get("data") or {}).get("item")
            if not isinstance(item, dict) or "score" not in item:
                continue
            ident = " ".join(
                str(x) for x in (
                    (comp.get("meta") or {}).get("componentName"),
                    comp.get("type"), comp.get("title"), item.get("type"),
                ) if x
            ).lower()
            if critic is None and "critic" in ident:
                critic = item
            elif user is None and "user" in ident:
                user = item
        if critic is None and user is None and len(components) > 8:
            critic = (components[6].get("data") or {}).get("item")
            user = (components[8].get("data") or {}).get("item")
        return critic, user

    def search(self, game_name: str, max_n: int = 1) -> list[MetacriticType]:
        """
        Searches for games by name and retrieves details.
        Returns a list of MetacriticType objects.
        """
        # Find name of the game in the Metacritic API
        find_data = self.http.get_json(
            f"https://backend.metacritic.com/finder/metacritic/search/{quote(game_name)}/web",
            params={"apiKey": self.api_key, "limit": max_n + 5, "offset": 0},
        )

        # No result found
        if "data" not in find_data:
            return []

        metacritic_results = []
        for item in find_data["data"]["items"]:
            if item["type"] != "game-title":
                continue
            # Match between names
            name_ratio = SequenceMatcher(None, item["title"], game_name).ratio()

            # Get game metadata
            game_slug = item["slug"]
            game_data = self.http.get_json(
                f"https://backend.metacritic.com/composer/metacritic/pages/games/{game_slug}/web",
                params={"contentOnly": "true"},
            )
            components = game_data.get("components") or []
            if not components or "data" not in components[0]:
                continue
            critic_item, user_item = self._find_score_items(components)
            if critic_item is None and user_item is None:
                continue

            metadata = components[0]["data"]["item"]
            metacritic_obj = MetacriticType(
                name=metadata.get("title"),
                release=metadata.get("releaseDate", ""),
                developers=[c["name"] for c in metadata.get("production", {}).get("companies", []) if c["typeName"] == "Developer"],
                publishers=[c["name"] for c in metadata.get("production", {}).get("companies", []) if c["typeName"] == "Publisher"],
                genres=[g["name"] for g in metadata.get("genres", [])],
                platforms=[p["name"] for p in metadata.get("platforms", [])],
                metacritic_rating=self._score_ratio(critic_item) if critic_item else 0.0,
                user_rating=self._score_ratio(user_item) if user_item else 0.0,
                slug=game_slug,
            )
            metacritic_results.append((metacritic_obj, name_ratio))

        # Sort by ratio of the name, return the top max_n
        sorted_data = sorted(metacritic_results, key=lambda x: x[1], reverse=True)
        return [obj for obj, ratio in sorted_data][:max_n]
