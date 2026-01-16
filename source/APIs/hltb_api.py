from howlongtobeatpy import HowLongToBeat
from APIs.api_types import HLTBType


class HLTB:
    def __init__(self):
        self.client = HowLongToBeat()

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
                game_name=result.game_name,
                game_type=result.game_type,
                review_score=result.review_score,
                profile_platforms=result.profile_platforms,
                release_world=result.release_world,
                main_story=result.main_story,
                main_extra=result.main_extra,
                completionist=result.completionist,
            )
            hltb_results.append(hltb_obj)

        return hltb_results
