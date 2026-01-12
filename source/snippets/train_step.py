import d3rlpy
import numpy as np

# 1. Load your Data
# observations: Array of shape (N, vector_dim) -> The User States
# actions: Array of shape (N, vector_dim) -> The Game Vectors played
# rewards: Array of shape (N, 1) -> +1/-1 values
# terminals: Array of shape (N, 1) -> 1 if session ended, else 0

dataset = d3rlpy.dataset.MDPDataset(
    observations=user_state_vectors,
    actions=game_vectors_played,
    rewards=rewards,
    terminals=terminals,
)

# 2. Train the Offline RL Algorithm (e.g., CQL or IQL)
# We use 'continuous' because we are recommending Vectors, not IDs
cql = d3rlpy.algos.CQL(
    action_scaler="min_max",
    gamma=0,
    actor_learning_rate=1e-4,
    critic_learning_rate=3e-4,
    use_gpu=True,
)
cql.fit(dataset, n_steps=10000)

# 3. Save Model
