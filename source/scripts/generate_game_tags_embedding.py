import pickle
import ast
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import lil_matrix, csr_matrix

N_COMPONENTS = 32  # testar 16, 32, 64

# Load games dataset
df = pd.read_csv("./data/games.csv")
df = df[["name", "genres", "keywords"]].dropna(subset=["name"])
df = df.drop_duplicates(subset=["name"])


def parse_tags(s):
    """Parse stringified list, lowercase, strip, drop empties."""
    if pd.isna(s):
        return []
    try:
        items = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []
    return [t.strip().lower() for t in items if isinstance(t, str) and t.strip()]


# Combine genres + keywords into a single deduped tag set per game
df["tags"] = df.apply(
    lambda r: list(set(parse_tags(r["genres"]) + parse_tags(r["keywords"]))),
    axis=1,
)

# Drop games with no tags (would be a zero vector, breaks cosine similarity)
df = df[df["tags"].map(len) > 0].reset_index(drop=True)

# Build vocabulary
all_tags = sorted({tag for tags in df["tags"] for tag in tags})
tag_to_idx = {tag: i for i, tag in enumerate(all_tags)}
print(f"Vocab size: {len(all_tags)} unique tags across {len(df)} games")

# Build sparse multi-hot matrix (games x tags)
matrix = lil_matrix((len(df), len(all_tags)), dtype=np.float32)
for row_idx, tags in enumerate(df["tags"].values):
    for tag in tags:
        matrix[row_idx, tag_to_idx[tag]] = 1.0
matrix = csr_matrix(matrix)

# Reduce dimensionality with TruncatedSVD
svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
embeddings = svd.fit_transform(matrix)
print(f"Embedding shape: {embeddings.shape}")
print(f"Explained variance sum: {svd.explained_variance_ratio_.sum():.4f}")

# Map back to game names
game_tag_dict = {name: emb for name, emb in zip(df["name"].values, embeddings)}

# Save
with open("./data/game_tags_embeddings.pkl", "wb") as f:
    pickle.dump(game_tag_dict, f)

print("Pickle saved to data/game_tags_embeddings.pkl")
