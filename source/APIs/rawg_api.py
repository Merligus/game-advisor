import os

from dotenv import load_dotenv

from APIs.api_types import RAWGType
from APIs.http_client import HttpClient

BASE = "https://api.rawg.io/api"


class RAWG:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("RAWG_API_KEY")
        self.http = HttpClient()

    def _parse_details(self, game_data: dict) -> RAWGType:
        return RAWGType(
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
            cover_url=[game_data.get("background_image")] if game_data.get("background_image") else [],
        )

    def get_by_id(self, game_id: int) -> RAWGType:
        """Full details for a known RAWG game id (used by the discovery pass so
        a discovered game doesn't need a second name search)."""
        return self._parse_details(self.http.get_json(f"{BASE}/games/{game_id}", params={"key": self.api_key}))

    def search(self, game_name: str, max_n: int = 1) -> list[RAWGType]:
        """
        Searches for games by name and retrieves details.
        Returns a list of RAWGType objects.
        """
        data = self.http.get_json(
            f"{BASE}/games",
            params={"search": game_name, "search_precise": "true", "key": self.api_key},
        )
        search_results = data.get("results", [])
        return [self.get_by_id(g["id"]) for g in search_results[:max_n] if g.get("id")]

    def list_recent(self, start_date: str, end_date: str, max_results: int = 100, min_added: int = 50) -> list[dict]:
        """Discovery: games released in [start_date, end_date] (YYYY-MM-DD),
        ordered by popularity (RAWG 'added' count, descending). Returns light
        dicts {name, released, added, id} — call get_by_id for full details.
        `min_added` gates out shovelware; paging stops early once results drop
        below it (the ordering is monotonic).
        """
        out, page = [], 1
        while len(out) < max_results:
            data = self.http.get_json(
                f"{BASE}/games",
                params={"key": self.api_key, "dates": f"{start_date},{end_date}",
                        "ordering": "-added", "page_size": 40, "page": page},
            )
            results = data.get("results", [])
            if not results:
                break
            for g in results:
                if (g.get("added") or 0) < min_added:
                    return out
                out.append({"name": g.get("name"), "released": g.get("released"),
                            "added": g.get("added"), "id": g.get("id")})
                if len(out) >= max_results:
                    return out
            if not data.get("next"):
                break
            page += 1
        return out
