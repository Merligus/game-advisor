import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from APIs.hltb_api import HLTB
from APIs.igdb_api import IGDB
from APIs.rawg_api import RAWG
from APIs.gamespot_api import Gamespot
from APIs.api_types import GameType
import pandas

if __name__ == "__main__":
    # APIs modules
    rawg = RAWG()
    igdb = IGDB()
    hltb = HLTB()
    gamespot = Gamespot()

    # aggregation list
    game_list: list[GameType] = []

    game_name = "Clair Obscur"

    # query
    results_rawg = rawg.search(game_name, max_n=1)
    results_igdb = igdb.search(game_name, max_n=1)
    results_hltb = hltb.search(game_name, max_n=1)
    results_gamespot = gamespot.search(game_name, max_n=1)

    for game in results_rawg:
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

    for game in results_igdb:
        print("\n--- Game Found ---")
        print(f"ID: {game.id}")
        print(f"Name: {game.name}")
        print(f"Game Modes: {game.game_modes}")
        print(f"Game Type: {game.game_type}")
        print(f"Keywords: {game.keywords}")
        print(f"Language Supports: {game.language_supports}")
        print(f"Platforms: {game.platforms}")
        print(f"Player Perspectives: {game.player_perspectives}")
        print(f"Themes: {game.themes}")
        print(f"Rating: {game.rating}")
        print(f"First Release Date: {game.first_release_date}")
        print(f"Genres: {game.genres}")
        print(f"Cover URL: {game.cover_url}")
        print(f"Summary: {game.summary}")

    for game in results_hltb:
        print("\n--- Game Found ---")
        print(f"Game Name: {game.game_name}")
        print(f"Game Type: {game.game_type}")
        print(f"Review Score: {game.review_score}")
        print(f"Profile Platforms: {game.profile_platforms}")
        print(f"Release World: {game.release_world}")
        print(f"Main Story: {game.main_story}")
        print(f"Main + Extra: {game.main_extra}")
        print(f"Completionist: {game.completionist}")

    for game in results_gamespot:
        print("\n--- Game Found ---")
        print(f"ID: {game.id}")
        print(f"Name: {game.name}")
        print(f"Themes: {game.themes}")
        print(f"Release_date: {game.release_date}")
        print(f"Genres: {game.genres}")
        print(f"Cover_url: {game.cover_url}")
        print(f"Description: {game.description}")
