import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import pickle

N = 8

# Load user reviews dataset
df = pd.read_csv("./data/metacritic_reviews.csv")

# Create a Pivot Table (Sparse Matrix)
# Rows = Users, Columns = Games, Values = Ratings
# Using sparse matrix because a dense pivot table use a lot of RAM
user_item_matrix = df.pivot_table(index="game_name", columns="author", values="score").fillna(0)

# Convert to sparse format for efficiency
sparse_matrix = csr_matrix(user_item_matrix.values)

# Apply SVD (Matrix Factorization)
# Compress the game info into N numbers
svd = TruncatedSVD(n_components=N, random_state=42)
svd.fit(sparse_matrix.T)

# Extract the Latent Vectors
# This matrix has shape (Number_of_Games, N)
item_vectors = svd.components_.T
print(item_vectors.shape)

# Map back to Game Titles
game_titles = user_item_matrix.index
game_embedding_dict = {title: vector for title, vector in zip(game_titles, item_vectors)}

# Save to test similar games
with open("./data/games_with_embeddings.pkl", "wb") as f:
    pickle.dump(game_embedding_dict, f)

print("Pickle saved to data/games_with_embeddings.pkl")

# # Save to Game Database
# # Load master game list (metadata)
# games_metadata_df = pd.read_csv("master_game_list.csv")

# # Create a new column 'svd_vector' by mapping the title
# games_metadata_df["svd_vector"] = games_metadata_df["title"].map(game_embedding_dict)

# # Filter out games that didn't have enough history to get a vector
# games_metadata_df = games_metadata_df.dropna(subset=["svd_vector"])

# # Save this for the App
# games_metadata_df.to_pickle("games_db_with_embeddings.pkl")
