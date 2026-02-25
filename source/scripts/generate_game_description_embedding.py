import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load games info dataset
df = pd.read_csv("./data/games.csv")
df_desc = df[["name", "description"]].dropna()

# Generate embeddings for each description
model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
game_desc_dict = {}
for index, row in tqdm(df_desc.iterrows(), total=len(df_desc), desc="Encoding descriptions"):
    game_desc_dict[row["name"]] = model.encode(row["description"])
    
# Save to test similar games
with open("./data/game_descriptions_embeddings.pkl", "wb") as f:
    pickle.dump(game_desc_dict, f)

print("Pickle saved to data/game_descriptions_embeddings.pkl")

# # Compute embeddings for both lists
# embeddings1 = model.encode(sentences1)
# embeddings2 = model.encode(sentences2)

# # Compute cosine similarities
# similarities = model.similarity(embeddings1, embeddings2)

# query_embedding = model.encode("How big is London")
# passage_embeddings = model.encode([
#     "London is known for its financial district",
#     "London has 9,787,426 inhabitants at the 2011 census",
#     "The United Kingdom is the fourth largest exporter of goods in the world",
# ])

# similarity = model.similarity(query_embedding, passage_embeddings)

# # Output the pairs with their score
# for idx_i, sentence1 in enumerate(sentences1):
#     print(sentence1)
#     for idx_j, sentence2 in enumerate(sentences2):
#         print(f" - {sentence2: <30}: {similarities[idx_i][idx_j]:.4f}")
