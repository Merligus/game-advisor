"""Train continuous-action IQL on the offline MDP dataset.

Loads Stage 2 artifacts + the combined embedding (Stage 1), splits each
user's reviews (sorted by date asc) into train (first 90%) + holdout
(last 10%) — but only carves a holdout when the user has at least
HOLDOUT_MIN_REVIEWS reviews; otherwise their full trajectory goes to
training. Trains d3rlpy IQL with gamma=0.2 (short-horizon recommendation,
per ideia.txt:107). Off-policy-evaluates with FQE on the holdout and
prints a cold-start top-5 sanity check.

Why IQL and not CQL (the original plan said CQL):
  d3rlpy's CQL implementation does `math.log(0.5**action_size)` to compute
  the log-density of uniform random actions over [-1, 1]^d. With our
  action_dim=1584, `0.5**1584` underflows to 0.0 in float64 and the call
  raises `ValueError: math domain error`. The stable form
  `1584 * math.log(0.5) = -1098` is mathematically correct, but plugging
  that into CQL's conservative loss (logsumexp over data + policy + random
  importance-weighted Q-values) lets the random term — offset by +1098 —
  dominate the in-distribution Q signal, so the conservative regularization
  is functionally degenerate at this action dim. IQL provides the same
  offline-safety guarantee (expectile-clipped value learning + AWR-style
  policy extraction) without that pathology, and is the modern offline-RL
  default for continuous high-dim actions.

Action space (follow-up A): the policy predicts in a low-dim PCA space, not the
full 1584-d embedding. The *state* stays the full 1584-d running mean of played
games (max signal), but the *action* target is Z = PCA(E) (D_ACT dims, from
`build_action_pca.py`). Regressing a 128-d action via AWR collapses far less
toward the centroid than a 1584-d one. Inference matches the predicted action
against normalized Z.

Saves:
  data/policy.pt          TorchScript policy (input dim 1584, output dim D_ACT)
  data/training_log.json  steps, FQE initial-state value, timings, seeds
  d3rlpy_logs/<exp>/      d3rlpy's per-epoch metrics + checkpoints (gitignored)

Reconstructs `actions = Z[action_row_idx]` (observations come straight from the
Stage 2 .npz; only the action matrix swapped from E to Z).

Requires d3rlpy + torch. Run from project root.
"""

import json
import random
import time

import numpy as np
import pandas as pd
import torch

import d3rlpy
from d3rlpy.algos import IQLConfig
from d3rlpy.dataset import MDPDataset
from d3rlpy.metrics import InitialStateValueEstimationEvaluator
from d3rlpy.ope import FQE, FQEConfig
from d3rlpy.preprocessing import MinMaxActionScaler

# -----------------------------------------------------------------------
# Knobs
# -----------------------------------------------------------------------
SEED = 42
N_STEPS_IQL = 50_000
N_STEPS_FQE = 10_000
HOLDOUT_FRAC = 0.1
HOLDOUT_MIN_REVIEWS = 5
BATCH_SIZE = 256
GAMMA = 0.2
OBS_DIM = 1584  # state dimension (full combined embedding)

# -----------------------------------------------------------------------
# 1. Reproducibility
# -----------------------------------------------------------------------
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
d3rlpy.seed(SEED)

device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
print(f"Device: {device}")
if device.startswith("cuda"):
    print(f"  GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# -----------------------------------------------------------------------
# 2. Load Stage 1 + Stage 2 artifacts
# -----------------------------------------------------------------------
print("\nLoading artifacts...")
data = np.load("./data/mdp_dataset.npz")
observations = data["observations"]
action_row_idx = data["action_row_idx"]
rewards = data["rewards"].astype(np.float32)
terminals = data["terminals"].astype(np.float32)

# Z = PCA-reduced action target (build_action_pca.py). Actions are reduced;
# observations stay full-dim (built from E in Stage 2).
Z = np.load("./data/game_actions_reduced.npy")
index_df = pd.read_pickle("./data/game_embeddings_index.pkl")
assert observations.shape[1] == OBS_DIM, f"unexpected obs dim {observations.shape[1]}"
ACTION_DIM = Z.shape[1]

actions = Z[action_row_idx].astype(np.float32)
print(f"  observations: {observations.shape}, dtype={observations.dtype}")
print(f"  actions:      {actions.shape} (reconstructed from Z[action_row_idx], reduced action space)")
print(f"  rewards:      {rewards.shape}")
print(f"  terminals:    {terminals.shape}  (sum={int(terminals.sum())})")

# -----------------------------------------------------------------------
# 3. Per-user train/holdout split via terminal positions
# -----------------------------------------------------------------------
end_indices = np.where(terminals == 1.0)[0]
start_indices = np.concatenate([[0], end_indices[:-1] + 1])
N_users = len(end_indices)
print(f"\nN_users = {N_users}")

train_mask = np.ones(len(observations), dtype=bool)
train_terminals = terminals.copy()

n_users_split = 0
n_holdout_total = 0
for start, end in zip(start_indices, end_indices):
    n_reviews = end - start + 1
    if n_reviews >= HOLDOUT_MIN_REVIEWS:
        k_holdout = max(1, int(round(n_reviews * HOLDOUT_FRAC)))
        holdout_start = end - k_holdout + 1
        train_mask[holdout_start : end + 1] = False
        # Re-mark training terminal: last training review of this user
        train_terminals[end] = 0.0
        train_terminals[holdout_start - 1] = 1.0
        n_users_split += 1
        n_holdout_total += k_holdout

train_obs = observations[train_mask]
train_act = actions[train_mask]
train_rew = rewards[train_mask]
train_term = train_terminals[train_mask]

holdout_obs = observations[~train_mask]
holdout_act = actions[~train_mask]
holdout_rew = rewards[~train_mask]
holdout_term = terminals[~train_mask]

print(f"  users with holdout carved: {n_users_split}/{N_users}")
print(f"  train transitions:   {len(train_obs)}   terminals={int(train_term.sum())}")
print(f"  holdout transitions: {len(holdout_obs)}   terminals={int(holdout_term.sum())}")

# -----------------------------------------------------------------------
# 4. d3rlpy datasets
# -----------------------------------------------------------------------
train_dataset = MDPDataset(
    observations=train_obs,
    actions=train_act,
    rewards=train_rew,
    terminals=train_term,
)
holdout_dataset = MDPDataset(
    observations=holdout_obs,
    actions=holdout_act,
    rewards=holdout_rew,
    terminals=holdout_term,
)

# -----------------------------------------------------------------------
# 5. Train IQL
# -----------------------------------------------------------------------
iql = IQLConfig(
    batch_size=BATCH_SIZE,
    gamma=GAMMA,
    action_scaler=MinMaxActionScaler(),
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    # expectile=0.7, weight_temp=3.0, max_weight=100 — defaults
).create(device=device)

print(f"\nTraining IQL for {N_STEPS_IQL} steps (gamma={GAMMA}, batch={BATCH_SIZE})...")
t0 = time.time()
iql.fit(
    train_dataset,
    n_steps=N_STEPS_IQL,
    n_steps_per_epoch=10_000,
    experiment_name="iql_continuous",
    show_progress=True,
)
train_time_iql = time.time() - t0
print(f"  IQL training took {train_time_iql:.1f}s")

# -----------------------------------------------------------------------
# 6. Save TorchScript policy
# -----------------------------------------------------------------------
iql.save_policy("./data/policy.pt")
print("  Saved TorchScript policy to ./data/policy.pt")

# -----------------------------------------------------------------------
# 7. FQE off-policy evaluation on holdout
# -----------------------------------------------------------------------
print(f"\nTraining FQE for {N_STEPS_FQE} steps on holdout...")
fqe = FQE(algo=iql, config=FQEConfig(), device=device)
t0 = time.time()
fqe.fit(
    holdout_dataset,
    n_steps=N_STEPS_FQE,
    n_steps_per_epoch=2_000,
    experiment_name="fqe_continuous",
    show_progress=True,
)
train_time_fqe = time.time() - t0
print(f"  FQE training took {train_time_fqe:.1f}s")

evaluator = InitialStateValueEstimationEvaluator()
fqe_value = float(evaluator(fqe, holdout_dataset))
print(f"  FQE initial-state value estimate (on holdout): {fqe_value:.4f}")

# -----------------------------------------------------------------------
# 8. Sanity check: cold-start state -> top-5 nearest games
# -----------------------------------------------------------------------
print("\nSanity check — cold-start state, top-5 by cosine to predicted action (Z space):")
cold_state = np.zeros((1, OBS_DIM), dtype=np.float32)
predicted_action = iql.predict(cold_state)[0]
Z_norm = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
pred_norm = predicted_action / max(float(np.linalg.norm(predicted_action)), 1e-12)
sims = Z_norm @ pred_norm
names = index_df["name"].values
for k, idx in enumerate(np.argsort(-sims)[:5]):
    print(f"  {k+1}. {sims[idx]:.4f}  {names[idx]}")

# -----------------------------------------------------------------------
# 9. Training log
# -----------------------------------------------------------------------
log = {
    "algo": "IQL",
    "seed": SEED,
    "device": device,
    "n_steps_iql": N_STEPS_IQL,
    "n_steps_fqe": N_STEPS_FQE,
    "batch_size": BATCH_SIZE,
    "gamma": GAMMA,
    "obs_dim": OBS_DIM,
    "action_dim": int(ACTION_DIM),
    "holdout_frac": HOLDOUT_FRAC,
    "holdout_min_reviews": HOLDOUT_MIN_REVIEWS,
    "n_users_total": int(N_users),
    "n_users_with_holdout": int(n_users_split),
    "train_transitions": int(len(train_obs)),
    "holdout_transitions": int(len(holdout_obs)),
    "fqe_initial_state_value": fqe_value,
    "train_time_iql_seconds": train_time_iql,
    "train_time_fqe_seconds": train_time_fqe,
}
with open("./data/training_log.json", "w") as fh:
    json.dump(log, fh, indent=2)
print("\nSaved training log to ./data/training_log.json")

print("\nNext: open source/scripts/test_policy.ipynb (Stage 3.5) to eyeball recs across user profiles.")
