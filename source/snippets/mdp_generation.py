import pandas as pd
import numpy as np
import d3rlpy
from sklearn.preprocessing import MultiLabelBinarizer

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Assuming you have these two files:
# reviews.csv -> Columns: 'user_id', 'game_id', 'rating', 'timestamp' (optional)
# games.csv   -> Columns: 'game_id', 'genres' (e.g., "Action|RPG|Indie")

GAME_VECTOR_DIM = 20  # How many dimensions for our Game Embeddings?


def main():
    print("--- 1. Loading Data ---")
    # MOCK DATA GENERATION (Replace this with pd.read_csv('reviews.csv'))
    # We simulate a "Reviews Line Dataset"
    reviews_df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "game_id": [101, 102, 103, 101, 104],
            "rating": [8.0, 4.0, 9.0, 7.0, 10.0],  # Raw 0-10
            "timestamp": [100, 101, 102, 200, 201],  # Optional
        }
    )

    # MOCK GAME DATA (Replace with pd.read_csv('games.csv'))
    games_df = pd.DataFrame(
        {
            "game_id": [101, 102, 103, 104],
            "genres": [
                ["Action", "RPG"],
                ["Strategy"],
                ["Action", "Souls-like"],
                ["RPG", "Turn-Based"],
            ],
        }
    )

    print("--- 2. Vectorizing Games (The 'Action' Space) ---")
    # We need to turn "Action|RPG" into a vector like [1, 0, 1, 0...]
    mlb = MultiLabelBinarizer()
    genre_vectors = mlb.fit_transform(games_df["genres"])

    # Map GameID -> Vector
    # Create a dictionary for fast lookup: {101: [1,0,1...], 102: [0,1,0...]}
    game_vector_map = {gid: vec for gid, vec in zip(games_df["game_id"], genre_vectors)}

    # Store dimensions for d3rlpy config later
    action_size = len(genre_vectors[0])
    print(f"Game Vector Size: {action_size} dimensions")

    print("--- 3. processing User Sessions ---")
    # Sort by User and Time (Critical for RL!)
    # If you don't have time, use: reviews_df.sample(frac=1).sort_values('user_id')
    if "timestamp" in reviews_df.columns:
        reviews_df = reviews_df.sort_values(by=["user_id", "timestamp"])
    else:
        reviews_df = reviews_df.sort_values(by=["user_id"])  # Arbitrary order

    # Initialize lists to hold the MDP data
    observations = []  # State (User History)
    actions = []  # Action (Game Vector)
    rewards = []  # Reward (Normalized Rating)
    terminals = []  # Done Flag

    # Group by User to create "Episodes"
    grouped = reviews_df.groupby("user_id")

    for user_id, group in grouped:

        # Initialize User State (e.g., Zero vector at start of session)
        # "State" = Average of games played so far
        current_user_state = np.zeros(action_size)

        # Count for averaging
        games_played_count = 0

        # Loop through this user's timeline
        for index, row in group.iterrows():
            game_id = row["game_id"]
            raw_rating = row["rating"]

            # 1. GET ACTION (The Game Vector)
            if game_id not in game_vector_map:
                continue  # Skip games we don't have metadata for

            game_vec = game_vector_map[game_id]

            # 2. CALCULATE REWARD (Normalize 0-10 to -1 to 1)
            # Using the logic we discussed earlier
            norm_reward = (raw_rating - 5.0) / 5.0

            # 3. APPEND TO BUFFERS
            observations.append(
                current_user_state.copy()
            )  # Record state BEFORE this action
            actions.append(game_vec)
            rewards.append([norm_reward])  # d3rlpy expects shape (N, 1)

            # 4. UPDATE STATE (Transition logic)
            # New State = Old State + This Game (Running Average)
            games_played_count += 1
            # Simple average update formula
            current_user_state = (
                current_user_state * (games_played_count - 1) + game_vec
            ) / games_played_count

            # 5. HANDLE TERMINAL
            # Is this the last review for this user?
            is_last_step = index == group.index[-1]
            terminals.append([1.0 if is_last_step else 0.0])

    print("--- 4. Finalizing Arrays ---")
    # Convert to Numpy
    obs_arr = np.array(observations, dtype=np.float32)
    act_arr = np.array(actions, dtype=np.float32)
    rew_arr = np.array(rewards, dtype=np.float32)
    term_arr = np.array(terminals, dtype=np.float32)

    print(f"Dataset Shape: {obs_arr.shape[0]} transitions")
    print(f"Observation Shape: {obs_arr.shape}")
    print(f"Action Shape: {act_arr.shape}")

    print("--- 5. Saving for d3rlpy ---")
    # Save as an MDPDataset object or raw files
    dataset = d3rlpy.dataset.MDPDataset(
        observations=obs_arr,
        actions=act_arr,
        rewards=rew_arr,
        terminals=term_arr,
    )

    # Dump to disk (optional)
    dataset.dump("game_recommender_data.h5")
    print("Success! 'game_recommender_data.h5' created.")

    # --- EXAMPLE TRAINING ---
    # cql = d3rlpy.algos.CQL(use_gpu=False)
    # cql.fit(dataset, n_steps=1000)


if __name__ == "__main__":
    main()
