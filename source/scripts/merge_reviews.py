import pandas as pd

# Load data/gamespot_reviews.csv in a pandas dataframe
gamespot_df = pd.read_csv("data/gamespot_reviews.csv")

# Load data/metacritic_reviews.csv in a pandas dataframe
metacritic_df = pd.read_csv("data/metacritic_reviews.csv")

# Show the duplicate lines head dataframe
duplicates_df = metacritic_df[metacritic_df.duplicated()]
print("Duplicate lines in metacritic_reviews:")
print(len(duplicates_df))
print(duplicates_df.head())

# Remove duplicate lines in metacritic_reviews dataframe
metacritic_df = metacritic_df.drop_duplicates()

# Standardize column names to match
# gamespot: authors,score,game_name,publish_date
# metacritic: author,score,game_name,date,type,platform
gamespot_df = gamespot_df.rename(columns={"authors": "author", "publish_date": "date"})

# Create a new dataframe that merges metacritic and gamespot reviews dataframes
reviews_df = pd.concat([gamespot_df, metacritic_df], ignore_index=True)

# Save this new dataframe in a file called data/reviews.csv
reviews_df.to_csv("data/reviews.csv", index=False)
