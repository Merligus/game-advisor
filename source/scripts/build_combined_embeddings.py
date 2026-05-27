"""Build the combined per-game embedding matrix E (dim 1584).

Concatenates (fixed order):
  1. title SBERT       (768) - L2-normalized
  2. description SBERT (768) - L2-normalized
  3. tags SVD          (32)  - L2-normalized
  4. collab SVD        (8)   - L2-normalized
  5. scalars block     (8)   - min-max scaled, divided by sqrt(8)

Scalars block: metacritic_rating, igdb_rating, rawg_rating, hltb_rating,
user_rating, main_story, main_extra, completionist. games.csv encodes
"no data" as 0.0 (not NaN), so we treat 0.0 as missing for imputation
purposes; time-to-beat columns are clipped at the 99th percentile before
min-max so a single 9000-hour outlier doesn't crush the scale.

Missing embedding blocks (a game lacks description/tags/collab) are
imputed with the per-block mean over present games. Games still missing
the title block are dropped (title is the anchor).

Outputs (data/):
  game_embeddings_matrix.npy   float32 (N, 1584)
  game_embeddings_index.pkl    DataFrame: name, row_idx, release_year,
                               platforms, language_supports, main_story,
                               main_extra, completionist
  embedding_scalers.pkl        {"scalars": per-column scaler params,
                                "block_means": per-block imputation means}

The index is stored as pickle (not parquet) to avoid a pyarrow dependency
in this Python-only project; downstream stages read via pd.read_pickle.

Run from project root.
"""

import ast
import pickle

import numpy as np
import pandas as pd

TITLE_D, DESC_D, TAGS_D, COLLAB_D, SCALAR_D = 768, 768, 32, 8, 8
E_DIM = TITLE_D + DESC_D + TAGS_D + COLLAB_D + SCALAR_D
assert E_DIM == 1584

SCALAR_COLS = [
    "metacritic_rating",
    "igdb_rating",
    "rawg_rating",
    "hltb_rating",
    "user_rating",
    "main_story",
    "main_extra",
    "completionist",
]
TTB_COLS = {"main_story", "main_extra", "completionist"}

# IGDB encodes TBA / unannounced games with far-future placeholder dates
# (this dataset clusters them at 2097-2099, with a clean gap after 2030).
# Clamp implausibly-distant release years to NaN so the year filter and the
# displayed year treat them as "unknown" rather than as a real 2098 release.
MAX_PLAUSIBLE_YEAR = 2035


def load_pkl(filename: str) -> dict:
    with open(f"./data/{filename}", "rb") as fh:
        return pickle.load(fh)


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def parse_list_column(s):
    if pd.isna(s):
        return []
    try:
        items = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []
    return [t for t in items if isinstance(t, str) and t.strip()]


# 1. Load inputs
print("Loading inputs...")
df = pd.read_csv("./data/games.csv")
df = df.dropna(subset=["name"]).drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)
print(f"  games.csv: {len(df)} unique games")

title_emb = load_pkl("game_title_embeddings.pkl")
desc_emb = load_pkl("game_descriptions_embeddings.pkl")
tags_emb = load_pkl("game_tags_embeddings.pkl")
collab_emb = load_pkl("games_with_embeddings.pkl")
print(f"  title: {len(title_emb)}  desc: {len(desc_emb)}  tags: {len(tags_emb)}  collab: {len(collab_emb)}")

# 2. Anchor: keep only games with a title embedding
df = df[df["name"].isin(title_emb)].reset_index(drop=True)
names = df["name"].tolist()
N = len(names)
print(f"\nAnchored to title embedding: N = {N}")

# 3. Build each embedding block + impute missing rows with block mean
block_means: dict[str, np.ndarray] = {}


def build_block(block_name: str, embedding_dict: dict, dim: int) -> np.ndarray:
    matrix = np.zeros((N, dim), dtype=np.float32)
    present = np.zeros(N, dtype=bool)
    for i, n in enumerate(names):
        if n in embedding_dict:
            matrix[i] = embedding_dict[n].astype(np.float32)
            present[i] = True
    n_present = int(present.sum())
    if n_present == 0:
        raise RuntimeError(f"No games have the {block_name} block")
    block_mean = matrix[present].mean(axis=0)
    block_means[block_name] = block_mean
    matrix[~present] = block_mean
    print(f"  {block_name}: present {n_present}/{N}, dim {dim}")
    return matrix


print("\nBuilding embedding blocks + imputing missing rows:")
title_block = build_block("title", title_emb, TITLE_D)
desc_block = build_block("description", desc_emb, DESC_D)
tags_block = build_block("tags", tags_emb, TAGS_D)
collab_block = build_block("collab", collab_emb, COLLAB_D)

# 4. L2-normalize each embedding block independently
print("\nL2-normalizing embedding blocks...")
title_block = l2_normalize(title_block)
desc_block = l2_normalize(desc_block)
tags_block = l2_normalize(tags_block)
collab_block = l2_normalize(collab_block)

# 5. Build scalar block
print("\nBuilding scalar block:")
scaler_params: dict[str, dict] = {}
scalar_block = np.zeros((N, SCALAR_D), dtype=np.float32)
for j, col in enumerate(SCALAR_COLS):
    raw = df[col].astype(np.float64).copy()
    valid_mask = raw > 0  # 0.0 encodes "no data"
    median = float(raw[valid_mask].median()) if valid_mask.any() else 0.0
    raw_imputed = raw.where(valid_mask, median).astype(np.float64)
    p99 = None
    if col in TTB_COLS and valid_mask.any():
        p99 = float(raw[valid_mask].quantile(0.99))
        raw_imputed = raw_imputed.clip(upper=p99)
    col_min = float(raw_imputed.min())
    col_max = float(raw_imputed.max())
    if col_max > col_min:
        scaled = (raw_imputed - col_min) / (col_max - col_min)
    else:
        scaled = raw_imputed * 0.0
    scalar_block[:, j] = scaled.astype(np.float32)
    scaler_params[col] = {"median": median, "p99": p99, "min": col_min, "max": col_max}
    print(f"  {col}: median={median:.4f}, p99={p99}, range=[{col_min:.4f}, {col_max:.4f}]")

scalar_block /= np.sqrt(SCALAR_D)

# 6. Concatenate -> E
E = np.concatenate([title_block, desc_block, tags_block, collab_block, scalar_block], axis=1).astype(np.float32)
assert E.shape == (N, E_DIM), f"unexpected E shape {E.shape}"
print(f"\nCombined embedding matrix E: {E.shape}, dtype={E.dtype}")

# 7. Build index DataFrame
release_year = pd.to_datetime(df["release"], errors="coerce").dt.year
# Implausibly-distant years (IGDB TBA placeholders) -> NaN (treated as unknown).
release_year = release_year.where(release_year <= MAX_PLAUSIBLE_YEAR)
index_df = pd.DataFrame(
    {
        "name": names,
        "row_idx": np.arange(N, dtype=np.int64),
        "release_year": release_year.values,
        "platforms": df["platforms"].apply(parse_list_column).tolist(),
        "language_supports": df["language_supports"].apply(parse_list_column).tolist(),
        "main_story": df["main_story"].values,
        "main_extra": df["main_extra"].values,
        "completionist": df["completionist"].values,
    }
)

# 8. Save artifacts
np.save("./data/game_embeddings_matrix.npy", E)
index_df.to_pickle("./data/game_embeddings_index.pkl")
with open("./data/embedding_scalers.pkl", "wb") as fh:
    pickle.dump({"scalars": scaler_params, "block_means": block_means}, fh)

print(f"\nSaved:")
print(f"  ./data/game_embeddings_matrix.npy   ({E.nbytes / 1e6:.1f} MB)")
print(f"  ./data/game_embeddings_index.pkl")
print(f"  ./data/embedding_scalers.pkl")

# 9. Sanity check: cosine top-5 against The Witcher 3: Wild Hunt
print("\nSanity check - cosine top-5 against 'The Witcher 3: Wild Hunt':")
target = "The Witcher 3: Wild Hunt"
if target not in names:
    candidates = [n for n in names if "witcher 3" in n.lower()]
    if candidates:
        target = candidates[0]
        print(f"  Using closest match: {target!r}")
    else:
        print(f"  '{target}' not found; skipping sanity check")
        target = None

if target is not None:
    target_idx = names.index(target)
    E_norm = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    sims = E_norm @ E_norm[target_idx]
    top_k = np.argsort(-sims)[:6]  # 6 to include self
    for k, idx in enumerate(top_k):
        marker = " (target)" if idx == target_idx else ""
        print(f"  {k}. {sims[idx]:.4f}  {names[idx]}{marker}")
