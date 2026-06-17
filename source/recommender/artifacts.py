"""Shared, cached loaders for the inference-time artifacts (Stages 4-6).

Each artifact is read once per process and memoized:
  embedding_matrix()             E, float32 (N, 1584) — state/profile/candidate space
  embedding_matrix_normalized()  L2-normalized E, for cosine scoring
  action_matrix()                Z, float32 (N, D_ACT) — PCA action space (policy output)
  action_matrix_normalized()     L2-normalized Z, for policy-mode cosine scoring
  index_frame()                  the game index DataFrame (Stage 1 output)
  name_to_row()                  {canonical name: row index into E/Z}
  policy()                       TorchScript IQL policy, loaded on CPU

Paths resolve relative to this file (<root>/source/app/artifacts.py ->
<root>/data), so importing modules don't depend on the current working dir.
The policy is loaded with map_location="cpu" because it was traced on GPU
during training but the HuggingFace Space (and most callers) run on CPU.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@lru_cache(maxsize=1)
def embedding_matrix() -> np.ndarray:
    return np.load(DATA_DIR / "game_embeddings_matrix.npy")


@lru_cache(maxsize=1)
def embedding_matrix_normalized() -> np.ndarray:
    E = embedding_matrix()
    norms = np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    return (E / norms).astype(np.float32)


@lru_cache(maxsize=1)
def action_matrix() -> np.ndarray:
    """Z = PCA-reduced per-game embedding (build_action_pca.py). This is the
    space the policy predicts in, so policy-mode reranking matches against it."""
    return np.load(DATA_DIR / "game_actions_reduced.npy")


@lru_cache(maxsize=1)
def action_matrix_normalized() -> np.ndarray:
    Z = action_matrix()
    norms = np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    return (Z / norms).astype(np.float32)


@lru_cache(maxsize=1)
def index_frame() -> pd.DataFrame:
    return pd.read_pickle(DATA_DIR / "game_embeddings_index.pkl")


@lru_cache(maxsize=1)
def name_to_row() -> dict:
    idx = index_frame()
    return dict(zip(idx["name"].values, idx["row_idx"].values))


@lru_cache(maxsize=1)
def games_metadata() -> pd.DataFrame:
    """games.csv deduped and indexed by canonical name, for cover/description
    lookups at recommendation time (the embedding index doesn't carry these)."""
    df = pd.read_csv(DATA_DIR / "games.csv")
    df = df.dropna(subset=["name"]).drop_duplicates(subset=["name"], keep="first")
    return df.set_index("name")


@lru_cache(maxsize=1)
def policy():
    import torch

    p = torch.jit.load(str(DATA_DIR / "policy.pt"), map_location="cpu")
    p.eval()
    return p
