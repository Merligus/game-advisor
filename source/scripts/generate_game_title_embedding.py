import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load games info dataset
df = pd.read_csv("./data/games.csv")
df_title = df[["name"]].dropna()

# Generate embeddings for each name
model = SentenceTransformer("all-mpnet-base-v2", device="cuda")
game_title_dict = {}
for index, row in tqdm(df_title.iterrows(), total=len(df_title), desc="Encoding titles"):
    game_title_dict[row["name"]] = model.encode(row["name"])

# Save to test similar games
with open("./data/game_title_embeddings.pkl", "wb") as f:
    pickle.dump(game_title_dict, f)

print("Pickle saved to data/game_title_embeddings.pkl")
