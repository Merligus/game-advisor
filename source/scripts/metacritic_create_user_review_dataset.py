import os
import pandas as pd
import requests
from dotenv import load_dotenv
from difflib import SequenceMatcher

# Load API key
load_dotenv()
METACRITIC_API_KEY = os.getenv("METACRITIC_API_KEY")

# Load gamespot reviews
df_gamespot = pd.read_csv("./data/gamespot_reviews.csv")
unique_games = df_gamespot["game_name"].unique()

# Initialize results dataframe
reviews_data = []
fetched_users = set()

# For each game, fetch Metacritic user reviews
for gn, game_name in enumerate(unique_games):
    # Find name of the game in the Metacritic API
    find_url = f"https://backend.metacritic.com/finder/metacritic/search/{game_name}/web?apiKey={METACRITIC_API_KEY}&limit=30&offset=0"
    find_response = requests.get(find_url)
    find_data = find_response.json()

    game_slug = ""
    best_ratio = 0.0
    for item in find_data["data"]["items"]:
        current_ratio = SequenceMatcher(None, item["title"], game_name).ratio()
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            game_slug = item["slug"]

    if len(game_slug) == 0:
        continue
    print(f"Game {game_name} {game_slug}")
    game_url = f"https://backend.metacritic.com/reviews/metacritic/user/games/{game_slug}/summary/web?apiKey={METACRITIC_API_KEY}"

    response = requests.get(game_url)
    game_data = response.json()

    # Concatenate positive, negative, neutral reviews
    reviews = (
        game_data["data"]["item"].get("positive", [])
        + game_data["data"]["item"].get("negative", [])
        + game_data["data"]["item"].get("neutral", [])
    )

    # For each review, get user_name and fetch their profile
    for rn, review in enumerate(reviews):
        user_name = review.get("author")

        if not user_name or user_name in fetched_users:
            continue

        fetched_users.add(user_name)

        # Check if user has 3+ reviews
        total_results = 1  # 1 to pass the while statement
        offset = 0
        limit = 100
        length_before = len(reviews_data)

        while True:
            user_url = f"https://backend.metacritic.com/reviews/metacritic/user/users/{user_name}/web?apiKey={METACRITIC_API_KEY}&filterByType=games&offset={offset}&limit={limit}&componentName=reviews&componentDisplayName=Profile+Reviews&componentType=ReviewListComponent&sort=date"
            user_response = requests.get(user_url)
            user_data = user_response.json()
            total_results = user_data["data"]["totalResults"]
            print(
                f"\t{user_name} has {total_results} reviews. currently in page {offset}"
            )

            if total_results >= 3:
                # Loop through user's reviews
                for item in user_data["data"]["items"]:
                    reviews_data.append(
                        {
                            "author": item.get("author"),
                            "score": item.get("score"),
                            "game_name": item.get("reviewedProduct", {}).get("title"),
                            "date": item.get("date"),
                            "type": item.get("reviewedProduct", {}).get("type"),
                            "platform": item.get("platform"),
                        }
                    )
                    print(
                        f"\t{reviews_data[-1]['author']}, {reviews_data[-1]['score']}, {reviews_data[-1]['game_name']}, {reviews_data[-1]['date']}"
                    )

                offset += limit
                if offset >= total_results:
                    break
            else:
                break
        # debug
        # if rn > 5:
        #     break
    # debug
    # if gn > 5:
    #     break

# Save results to CSV
df_reviews = pd.DataFrame(reviews_data)
df_reviews.to_csv("./data/metacritic_reviews.csv", index=False)
print(f"Saved {len(df_reviews)} reviews to ./data/metacritic_reviews.csv")
