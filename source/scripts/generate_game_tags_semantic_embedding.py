import pickle
import ast
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load games dataset
df = pd.read_csv("./data/games.csv")
df = df[["name", "genres", "keywords"]].dropna(subset=["name"])
df = df.drop_duplicates(subset=["name"])


def parse_tags(s):
    """Parse stringified list, strip, drop empties. Keep original case for semantic model."""
    if pd.isna(s):
        return []
    try:
        items = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []
    return [t.strip() for t in items if isinstance(t, str) and t.strip()]


# Combine genres + keywords into a single deduped list (preserve order roughly)
def build_tag_string(row):
    seen = set()
    tags = []
    for t in parse_tags(row["genres"]) + parse_tags(row["keywords"]):
        key = t.lower()
        if key not in seen:
            seen.add(key)
            tags.append(t)
    return ", ".join(tags)


df["tag_string"] = df.apply(build_tag_string, axis=1)
df = df[df["tag_string"].str.len() > 0].reset_index(drop=True)

# Encode with SentenceTransformer (same model used for descriptions)
model = SentenceTransformer("all-mpnet-base-v2", device="cuda")

game_tag_semantic_dict = {}
for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding tag strings"):
    game_tag_semantic_dict[row["name"]] = model.encode(row["tag_string"])

# Save to test similar games
with open("./data/game_tags_semantic_embeddings.pkl", "wb") as f:
    pickle.dump(game_tag_semantic_dict, f)

print("Pickle saved to data/game_tags_semantic_embeddings.pkl")
