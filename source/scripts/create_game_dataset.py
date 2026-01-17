import sys
from pathlib import Path
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))

from APIs.hltb_api import HLTB
from APIs.igdb_api import IGDB
from APIs.rawg_api import RAWG
from APIs.gamespot_api import Gamespot
from APIs.metacritic_api import Metacritic
from APIs.api_types import (
    RAWGType,
    IGDBType,
    HLTBType,
    GamespotType,
    MetacriticType,
    GameType,
)
import pandas as pd
from thefuzz import fuzz
from datetime import datetime
from time import sleep


def nameRatio(name1: str, name2: str) -> bool:
    return fuzz.ratio(name1, name2) / 100.0


def compareRelease(date1_s: str, date2_s: str) -> bool:
    format_pattern = "%Y-%m-%d"
    if date1_s is None or date2_s is None:
        return False
    if len(date1_s) == 0 or len(date2_s) == 0:
        return False
    date1 = datetime.strptime(date1_s, format_pattern)
    date2 = datetime.strptime(date2_s, format_pattern)
    return abs(date1.year - date2.year) <= 1


def getFirstString(str_list: list) -> str:
    for s in str_list:
        if s is not None and len(s) > 0:
            return s
    return ""


def getFirstFloat(float_list: list) -> float:
    for f in float_list:
        if f is not None and f > 0.0:
            return f
    return 0.0


def getFirstList(list_of_list: list) -> list:
    for l in list_of_list:
        if l is not None and len(l) > 0:
            return l
    return []


def getUnion(list_of_list: list) -> list:
    result = []
    for l in list_of_list:
        if l is not None:
            result += l
    return result


def showResults(api: str, results: GameType, game_name: str, game_release: str, debug: bool):
    if not debug:
        return
    print(f"\t-- {api} {len(results)}")
    game = results[0] if len(results) > 0 else GameType()
    print(f"\t\t{nameRatio(game.name, game_name)} {compareRelease(game.release, game_release)}")
    print(f"\t\tid: {game.id}")
    print(f"\t\tname: {game.name}")
    print(f"\t\trelease: {game.release}")
    print(f"\t\trawg_rating: {game.rawg_rating}")
    print(f"\t\tigdb_rating: {game.igdb_rating}")
    print(f"\t\thltb_rating: {game.hltb_rating}")
    print(f"\t\tmetacritic_rating: {game.metacritic_rating}")
    print(f"\t\tuser_rating: {game.user_rating}")
    print(f"\t\tplatforms: {game.platforms[:5]}")
    print(f"\t\tmain_story: {game.main_story}")
    print(f"\t\tmain_extra: {game.main_extra}")
    print(f"\t\tcompletionist: {game.completionist}")
    print(f"\t\tcover_url: {game.cover_url}")
    print(f"\t\tdevelopers: {game.developers[:5]}")
    print(f"\t\tpublishers: {game.publishers[:5]}")
    print(f"\t\tlanguage_supports: {game.language_supports[:5]}")
    print(f"\t\tgenres: {game.genres[:5]}")
    print(f"\t\tkeywords: {game.keywords[:5]}")
    print(f"\t\tdescription: {game.description[:10]}")


if __name__ == "__main__":
    # APIs modules
    rawg = RAWG()
    igdb = IGDB()
    hltb = HLTB()
    gamespot = Gamespot()
    metacritic = Metacritic()

    # Aggregation list
    game_list = []

    # Parameters
    goodRatio = 0.9
    minRatio = 0.65
    maxRetries = 3
    retryN = 0
    saveEveryNGames = 10
    # Rate limit handling
    baseDelay = 1  # seconds between API calls
    rateLimitBackoff = 60  # seconds to wait on rate limit
    # Debug
    debug = False
    maxGames = 2

    # Load gamespot reviews
    df_reviews = pd.read_csv("./data/reviews.csv")
    unique_games = [f"{game_name}" for game_name in df_reviews["game_name"].unique()]
    unique_games.sort()
    gi = 0
    while gi < len(unique_games):
        game_name = unique_games[gi]
        try:
            # Query and get the first result
            # Gamespot
            results_gamespot = gamespot.search(game_name, max_n=1)
            sleep(baseDelay)
            game_gamespot = results_gamespot[0] if len(results_gamespot) > 0 else GamespotType()
            game_gamespot = game_gamespot if nameRatio(game_gamespot.name, game_name) > minRatio else GamespotType()
            game_release = game_gamespot.release
            print(game_name, game_release)
            showResults("Gamespot", results_gamespot, game_name, game_release, debug)
            # RAWG
            results_rawg = rawg.search(game_name, max_n=1)
            sleep(baseDelay)
            game_rawg = results_rawg[0] if len(results_rawg) > 0 else RAWGType()
            game_rawg = game_rawg if nameRatio(game_rawg.name, game_name) > goodRatio or (nameRatio(game_rawg.name, game_name) > minRatio and compareRelease(game_rawg.release, game_release)) else RAWGType()
            showResults("RAWG", results_rawg, game_name, game_release, debug)
            # IGDB
            results_igdb = igdb.search(game_name, max_n=1)
            sleep(baseDelay)
            game_igdb = results_igdb[0] if len(results_igdb) > 0 else IGDBType()
            game_igdb = game_igdb if nameRatio(game_igdb.name, game_name) > goodRatio or (nameRatio(game_igdb.name, game_name) > minRatio and compareRelease(game_igdb.release, game_release)) else IGDBType()
            showResults("IGDB", results_igdb, game_name, game_release, debug)
            # HLTB
            results_hltb = hltb.search(game_name, max_n=1)
            sleep(baseDelay)
            game_hltb = results_hltb[0] if len(results_hltb) > 0 else HLTBType()
            game_hltb = game_hltb if nameRatio(game_hltb.name, game_name) > goodRatio or (nameRatio(game_hltb.name, game_name) > minRatio and compareRelease(game_hltb.release, game_release)) else HLTBType()
            showResults("HLTB", results_hltb, game_name, game_release, debug)
            # Metacritic
            results_metacritic = metacritic.search(game_name, max_n=1)
            sleep(baseDelay)
            game_metacritic = results_metacritic[0] if len(results_metacritic) > 0 else MetacriticType()
            game_metacritic = game_metacritic if nameRatio(game_metacritic.name, game_name) > goodRatio or (nameRatio(game_metacritic.name, game_name) > minRatio and compareRelease(game_metacritic.release, game_release)) else MetacriticType()
            showResults("Metacritic", results_metacritic, game_name, game_release, debug)

            game_obj = GameType(
                id=gi,
                name=getFirstString([game.name for game in [game_gamespot, game_rawg, game_igdb, game_hltb, game_metacritic]]),
                release=game_release,
                rawg_rating=game_rawg.rawg_rating,
                igdb_rating=game_igdb.igdb_rating,
                hltb_rating=game_hltb.hltb_rating,
                metacritic_rating=getFirstFloat([game.metacritic_rating for game in [game_metacritic, game_rawg]]),
                user_rating=game_metacritic.user_rating,
                platforms=getFirstList([game.platforms for game in [game_igdb, game_rawg, game_hltb, game_metacritic]]),
                main_story=getFirstFloat([game.main_story for game in [game_hltb, game_rawg]]),
                main_extra=game_hltb.main_extra,
                completionist=game_hltb.completionist,
                cover_url=getUnion([game.cover_url for game in [game_igdb, game_gamespot]]),
                developers=getFirstList([game.developers for game in [game_metacritic, game_rawg]]),
                publishers=getFirstList([game.publishers for game in [game_metacritic, game_rawg]]),
                description=getFirstString([game.description for game in [game_gamespot, game_rawg, game_igdb]]),
                language_supports=game_igdb.language_supports,
                genres=getUnion([game.genres for game in [game_rawg, game_igdb, game_gamespot, game_metacritic]]),
                keywords=getUnion([game.keywords for game in [game_rawg, game_igdb]] + [game_igdb.themes] + [game_igdb.game_modes] + [game_igdb.player_perspectives] + [[game_rawg.esrb_rating]] + [game_gamespot.themes]),
            )
            game_list.append(game_obj)
            gi += 1
            retryN = 0

            # Save every N games to not lose
            if gi % saveEveryNGames == 0:
                # Save results to CSV with append mode
                df_games = pd.DataFrame(game_list)
                df_games.to_csv("./data/games.csv", index=False)
                print(f"Saved {len(df_games)} games to ./data/games.csv")

            # Debug
            if debug:
                print(f"\tid: {game_obj.id}")
                print(f"\tname: {game_obj.name}")
                print(f"\trelease: {game_obj.release}")
                print(f"\trawg_rating: {game_obj.rawg_rating}")
                print(f"\tigdb_rating: {game_obj.igdb_rating}")
                print(f"\thltb_rating: {game_obj.hltb_rating}")
                print(f"\tmetacritic_rating: {game_obj.metacritic_rating}")
                print(f"\tuser_rating: {game_obj.user_rating}")
                print(f"\tplatforms: {game_obj.platforms[:5]}")
                print(f"\tmain_story: {game_obj.main_story}")
                print(f"\tmain_extra: {game_obj.main_extra}")
                print(f"\tcompletionist: {game_obj.completionist}")
                print(f"\tcover_url: {game_obj.cover_url}")
                print(f"\tdevelopers: {game_obj.developers[:5]}")
                print(f"\tpublishers: {game_obj.publishers[:5]}")
                print(f"\tlanguage_supports: {game_obj.language_supports[:5]}")
                print(f"\tgenres: {game_obj.genres[:5]}")
                print(f"\tkeywords: {game_obj.keywords[:5]}")
                print(f"\tdescription: {game_obj.description[:10]}")

                # Stop
                if gi == maxGames:
                    break
        except Exception as e:
            error_msg = str(e).lower()
            is_rate_limit = any(keyword in error_msg for keyword in ["rate limit", "429", "420", "too many requests", "quota"])

            print(50 * "*")
            print(50 * "-")
            if is_rate_limit:
                print(f"RATE LIMIT EXCEEDED - Waiting {rateLimitBackoff} seconds...")
                sleep(rateLimitBackoff)
            traceback.print_exc()
            print(50 * "-")
            print(50 * "*")

            # Wait
            sleep(3)
            # Continue if maxRetries reached
            retryN += 1
            if retryN >= maxRetries:
                gi += 1
                retryN = 0

    # Save results to CSV with append mode
    df_games = pd.DataFrame(game_list)
    df_games.to_csv("./data/games.csv", index=False)
    print(f"Saved {len(df_games)} games to ./data/games.csv")
