import requests
import pandas as pd
from dotenv import load_dotenv
import os
import xml.etree.ElementTree as ET


def load_env():
    load_dotenv()
    return os.getenv("GAMESPOT_API_KEY")


API_KEY = load_env()
url = f"https://www.gamespot.com/api/reviews/"

reviews_data = []
offset = 0
limit = 100

# Set headers for the request
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# Fetch all reviews with pagination
while True:
    params = {"offset": offset, "limit": limit, "api_key": API_KEY}
    response = requests.get(url, params=params, headers=headers)
    print(f"Status Code: {response.status_code}")

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        continue

    # Parse XML response
    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as e:
        print(f"XML Parse Error: {e}")
        print(f"Response: {response.text[:500]}")
        continue

    # Extract number of results
    number_of_total_results = int(root.findtext("number_of_total_results", 0))
    number_of_page_results = int(root.findtext("number_of_page_results", 0))

    # Extract reviews from this page
    for review in root.findall(".//review"):
        reviews_data.append(
            {
                "authors": review.findtext("authors"),
                "score": float(review.findtext("score", 0)),
                "game_name": review.findtext("game/name"),
                "publish_date": review.findtext("publish_date"),
            }
        )

    # Check if we've fetched all results
    offset += number_of_page_results
    if offset >= number_of_total_results:
        break

# Create DataFrame and save
df = pd.DataFrame(reviews_data)
df.to_csv("./data/gamespot_reviews.csv", index=False)
print(f"Saved {len(df)} reviews to gamespot_reviews.csv")
