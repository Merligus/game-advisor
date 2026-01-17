from howlongtobeatpy import HowLongToBeat
from APIs.api_types import HLTBType
from datetime import datetime


class HLTB:
    def __init__(self):
        self.client = HowLongToBeat()
        
        # Format pattern to get release date 2025
        self.format_pattern = "%Y"

    def search(self, game_name: str, max_n: int = 1) -> list[HLTBType]:
        """
        Searches for games by name on HowLongToBeat.
        Returns a list of HLTBType objects sorted by similarity.
        """
        results_list = self.client.search(game_name)

        if not results_list or len(results_list) == 0:
            return []

        # Sort by similarity and return top max_n results
        sorted_results = sorted(results_list, key=lambda x: x.similarity, reverse=True)[:max_n]

        hltb_results = []
        for result in sorted_results:
            hltb_obj = HLTBType(
                name=result.game_name,
                game_type=result.game_type,
                hltb_rating=float(result.review_score) / 100. if result.review_score else 0.0,
                platforms=result.profile_platforms,
                release=datetime.strptime(f'{result.release_world}', self.format_pattern).strftime("%Y-%m-%d"),
                main_story=result.main_story,
                main_extra=result.main_extra,
                completionist=result.completionist,
            )
            hltb_results.append(hltb_obj)

        return hltb_results
