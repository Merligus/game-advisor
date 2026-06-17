"""Follow-up A: build a low-dimensional action space via PCA on the embedding E.

The IQL actor collapses toward the embedding centroid partly because regressing
a 1584-d continuous action via advantage-weighted regression converges to a
near-constant conditional mean. Predicting in a much lower-dim space leaves far
less room for that collapse. This fits a PCA on the combined embedding matrix E
and stores the reduced per-game vectors Z (the new action target).

Pipeline change: the *state* the policy reads stays the full 1584-d running mean
of played games (maximum signal), but the *action* it predicts — and that we
match recommendations against — becomes the D_ACT-dim PCA projection Z.

Outputs (data/):
  game_actions_reduced.npy   float32 (N, D_ACT)  — Z = PCA(E); the action target
  action_pca.pkl             the fitted sklearn PCA (provenance / reproducibility)

Stage 3 (train_iql) trains actions = Z[action_row_idx]; inference matches the
policy's predicted action against normalized Z. E is unchanged, so the candidate
generator / profile rerank / state builder are unaffected.

Run from project root.
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

D_ACT = 128

print("Loading combined embedding E...")
E = np.load("./data/game_embeddings_matrix.npy")
index_df = pd.read_pickle("./data/game_embeddings_index.pkl")
print(f"  E: {E.shape}")

print(f"\nFitting PCA -> {D_ACT} components...")
pca = PCA(n_components=D_ACT, random_state=42)
Z = pca.fit_transform(E).astype(np.float32)
evr = pca.explained_variance_ratio_
print(f"  Z: {Z.shape}, dtype={Z.dtype}")
for d in (32, 64, 128):
    if d <= D_ACT:
        print(f"  cumulative explained variance @ {d:>3} comps: {evr[:d].sum():.4f}")

np.save("./data/game_actions_reduced.npy", Z)
with open("./data/action_pca.pkl", "wb") as fh:
    pickle.dump(pca, fh)
print("\nSaved:")
print(f"  ./data/game_actions_reduced.npy  ({Z.nbytes / 1e6:.1f} MB)")
print(f"  ./data/action_pca.pkl")

# Sanity: does the reduced space preserve neighborhoods? Top-5 for The Witcher 3.
print("\nSanity check - cosine top-5 in Z space against 'The Witcher 3: Wild Hunt':")
names = index_df["name"].values
target = "The Witcher 3: Wild Hunt"
if target in set(names):
    ti = list(names).index(target)
    Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    sims = Zn @ Zn[ti]
    for k, idx in enumerate(np.argsort(-sims)[:6]):
        marker = " (target)" if idx == ti else ""
        print(f"  {k}. {sims[idx]:.4f}  {names[idx]}{marker}")
