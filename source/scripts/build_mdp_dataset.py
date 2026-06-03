"""Build the offline-RL MDP dataset from reviews + the combined embedding.

Continuous-action formulation (see PLAN.md):
  state    = running average of past action vectors (zero vector at t=0, dim 1584)
  action   = E-vector of the rated game
  reward   = per-user z-scored review score, clipped to [-3, 3] and /3 -> [-1, 1]
             (see the reward block below for the rationale; replaces the old
             global (score - 5) / 5 which gave almost no advantage contrast)
  terminal = 1 on the user's last review (sorted by date asc), else 0

Drops reviews with null author/game_name or whose game isn't in the
embedding index. Per-user running mean is computed vectorized via
cumsum / arange rather than iterrows for speed.

Outputs (data/):
  mdp_dataset.npz  observations   (N, 1584) float32,
                   action_row_idx (N,)      int32,
                   rewards        (N,)      float32,
                   terminals      (N,)      float32
  mdp_meta.json    {N_transitions, N_users, action_dim, reward stats, ...}

We store `action_row_idx` (int32 pointers into the embedding matrix)
rather than materialized action vectors so the .npz stays small
(~1 GB instead of ~2.6 GB) and stays auto-synced if `E` is ever
rebuilt. Stage 3 reconstructs actions via `actions = E[action_row_idx]`
before passing to `d3rlpy.dataset.MDPDataset(...)`.

Saved as .npz (not d3rlpy's .h5) so Stage 2 has no d3rlpy dependency.

Run from project root.
"""

import json

import numpy as np
import pandas as pd
from tqdm import tqdm

ACTION_DIM = 1584

# 1. Load combined embedding + index
print("Loading combined embedding + index...")
E = np.load("./data/game_embeddings_matrix.npy")
index_df = pd.read_pickle("./data/game_embeddings_index.pkl")
assert E.shape[1] == ACTION_DIM, f"unexpected action dim {E.shape[1]}"
name_to_row = dict(zip(index_df["name"].values, index_df["row_idx"].values))
print(f"  E: {E.shape}, dtype={E.dtype}")
print(f"  index: {len(index_df)} games")

# 2. Load reviews + clean
print("\nLoading reviews.csv...")
df = pd.read_csv("./data/reviews.csv")
n0 = len(df)
df = df.dropna(subset=["author", "game_name", "score", "date"])
print(f"  raw: {n0} -> after dropna: {len(df)} ({n0 - len(df)} dropped)")

# Drop reviews whose game isn't in the embedding index
in_index = df["game_name"].isin(name_to_row)
print(f"  game_name in embedding index: {in_index.sum()}/{len(df)} ({in_index.mean()*100:.1f}%)")
df = df[in_index].copy()

# Parse date; drop unparseable. format="mixed" is required because
# gamespot reviews use 'YYYY-MM-DD HH:MM:SS' and metacritic reviews use
# 'YYYY-MM-DD' — without it, pandas 2.x infers one format and silently
# NaTs the other (we lost 94% of rows the first time).
df["date_parsed"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
n_before = len(df)
df = df.dropna(subset=["date_parsed"])
print(f"  date parsed: {n_before} -> {len(df)} ({n_before - len(df)} unparseable dropped)")

# 3. Sort by (author, date) ascending
df = df.sort_values(by=["author", "date_parsed"], kind="mergesort").reset_index(drop=True)
N = len(df)
N_users = df["author"].nunique()
print(f"\nTransitions: N = {N}, users = {N_users}")

# 4. Map game names -> E row indices (vectorized). int32 is plenty for our
# ~26K-game catalog and halves the on-disk size of this array.
action_row_idx = df["game_name"].map(name_to_row).to_numpy(dtype=np.int32)

# 5. Preallocate output arrays (no `actions` matrix — Stage 3 reconstructs
# it via E[action_row_idx])
observations = np.zeros((N, ACTION_DIM), dtype=np.float32)
terminals = np.zeros(N, dtype=np.float32)

# Reward = per-user z-scored review score, clipped to [-Z_CLIP, Z_CLIP] then
# rescaled to [-1, 1]. Centering each user's ratings on their own mean turns the
# signal into "did this user like this game *relative to their own taste*",
# which gives the offline-RL advantage estimate real contrast: ~42% of
# transitions become negative, versus only ~17% under the old global
# (score - 5) / 5 (mean +0.41). That flat, mostly-positive signal gave IQL's
# advantage-weighted policy extraction nothing to discriminate on, collapsing
# the actor toward the embedding centroid (measured: same recommendations from
# every state, predicted-action cosines bunched ~0.74). Users with <2 reviews
# (no std) or zero rating variance fall back to global mean/std centering so
# their absolute level vs the population is still encoded.
Z_CLIP = 3.0
global_mean = float(df["score"].mean())
global_std = float(df["score"].std())
user_mean = df.groupby("author")["score"].transform("mean")
user_std = df.groupby("author")["score"].transform("std")
z = (df["score"] - user_mean) / user_std
z = z.where(user_std.notna() & (user_std > 0), (df["score"] - global_mean) / global_std)
rewards = (np.clip(z.to_numpy(), -Z_CLIP, Z_CLIP) / Z_CLIP).astype(np.float32)

# 6. Per-user vectorized fill
# observations[i] for the j-th step of a user = mean of that user's previous (j-1) action vectors
# obs[0] within a user = zeros; obs[j] = cumsum(actions[:j]) / j  (for j >= 1)
print("\nBuilding per-user transitions (vectorized cumulative mean)...")
group_starts = df.groupby("author", sort=False).indices  # ordered dict-like of arrays

for author, idxs in tqdm(group_starts.items(), total=N_users, desc="users"):
    rows = action_row_idx[idxs]
    user_actions = E[rows]  # (N_user, ACTION_DIM) — only needed locally for the running mean
    # observations: shift-right of cumulative mean, with leading zero row
    if len(idxs) > 1:
        cumsum = np.cumsum(user_actions[:-1], axis=0, dtype=np.float32)
        denom = np.arange(1, len(idxs), dtype=np.float32)[:, None]
        observations[idxs[1:]] = (cumsum / denom).astype(np.float32)
    # observations[idxs[0]] stays at zeros (no history at first step)
    # terminal on last row
    terminals[idxs[-1]] = 1.0

assert int(terminals.sum()) == N_users, f"terminal sum {int(terminals.sum())} != N_users {N_users}"

# 7. Save artifacts
print("\nSaving artifacts (compressed)...")
np.savez_compressed(
    "./data/mdp_dataset.npz",
    observations=observations,
    action_row_idx=action_row_idx,
    rewards=rewards,
    terminals=terminals,
)

# Reward stats
reward_q = np.quantile(rewards, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
hist_bins = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.01]
hist_counts, _ = np.histogram(rewards, bins=hist_bins)
hist = {f"[{hist_bins[i]:.1f},{hist_bins[i+1]:.1f})": int(c) for i, c in enumerate(hist_counts)}

meta = {
    "N_transitions": int(N),
    "N_users": int(N_users),
    "action_dim": int(ACTION_DIM),
    "reward_quantiles": {q: float(reward_q[i]) for i, q in enumerate(["min", "p10", "p25", "p50", "p75", "p90", "max"])},
    "reward_histogram": hist,
    "terminal_sum": int(terminals.sum()),
}
with open("./data/mdp_meta.json", "w") as fh:
    json.dump(meta, fh, indent=2)

print(f"\nSaved:")
print(f"  ./data/mdp_dataset.npz")
print(f"  ./data/mdp_meta.json")

print("\nVerification:")
print(f"  observations.shape   = {observations.shape}, dtype={observations.dtype}")
print(f"  action_row_idx.shape = {action_row_idx.shape}, dtype={action_row_idx.dtype}")
print(f"    range: min={action_row_idx.min()}, max={action_row_idx.max()} (E has {E.shape[0]} rows)")
print(f"  rewards.shape        = {rewards.shape}, dtype={rewards.dtype}")
print(f"  terminals.shape      = {terminals.shape}, dtype={terminals.dtype}")
print(f"  reward quantiles  : {meta['reward_quantiles']}")
print(f"  reward histogram  : {meta['reward_histogram']}")
print(f"  terminal sum      : {meta['terminal_sum']} (should == N_users {N_users})")
