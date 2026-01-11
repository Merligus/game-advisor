import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

# 1. Load your Kaggle Interaction Data
# Assume columns: 'user_id', 'game_id', 'rating'
df = pd.read_csv("kaggle_video_game_reviews.csv")

# 2. Create a Pivot Table (Sparse Matrix)
# Rows = Users, Columns = Games, Values = Ratings
# We use a sparse matrix because a dense pivot table would crash your RAM
user_item_matrix = df.pivot_table(index='user_id', columns='game_title', values='rating').fillna(0)
# Convert to sparse format for efficiency
sparse_matrix = csr_matrix(user_item_matrix.values)

# 3. Apply SVD (Matrix Factorization)
# n_components=32 means we compress the game info into 32 numbers
svd = TruncatedSVD(n_components=32, random_state=42)
svd.fit(sparse_matrix.T) # Note: Transpose (.T) because we want Item vectors, not User vectors

# 4. Extract the Latent Vectors
# This matrix has shape (Number_of_Games, 32)
item_vectors = svd.components_.T 

# 5. Map back to Game Titles
game_titles = user_item_matrix.columns
game_embedding_dict = {title: vector for title, vector in zip(game_titles, item_vectors)}

# 6. Save to your Game Database
# Load your master game list (metadata)
games_metadata_df = pd.read_csv("master_game_list.csv")

# Create a new column 'svd_vector' by mapping the title
games_metadata_df['svd_vector'] = games_metadata_df['title'].map(game_embedding_dict)

# Filter out games that didn't have enough history to get a vector
games_metadata_df = games_metadata_df.dropna(subset=['svd_vector'])

# Save this for your App
games_metadata_df.to_pickle("games_db_with_embeddings.pkl")