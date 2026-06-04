<!--
HuggingFace Spaces reads the YAML front-matter below to configure the Space.
It also renders as the Space's README. On GitHub the block shows as a small
metadata table; that's expected for a repo that is also a HF Space.
-->
---
title: Game Advisor
emoji: 🎮
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.15.1
app_file: app.py
python_version: "3.11"
pinned: false
---

# Game Advisor

## Install

Install pytorch and other requirements.

- Change directory to `game-advisor/`;

- Create the environment in conda with python 3.11:
```bash
conda create -n GameAdvisor python=3.11 -y
conda activate GameAdvisor
```

```bash
pip install -r requirements.txt
```

- Create the env file with the following vars:
```bash
GAMESPOT_API_KEY=
RAWG_API_KEY=
IGDB_CLIENT_SECRET=
IGDB_CLIENT_ID=
```

## Run

### Create the user game review dataset

In the game-advisor folder, build the review tables (Gamespot, then Metacritic seeded from Gamespot's unique games, then merge into `data/reviews.csv`):

```bash
python source/scripts/gamespot_create_user_review_dataset.py
python source/scripts/metacritic_create_user_review_dataset.py
python source/scripts/merge_reviews.py
```

### Create the game dataset

```bash
python source/scripts/create_game_dataset.py
```

or

```bash
fish source/scripts/call_script_periodically.sh
```

### Collaborative Embedding

To retrieve the reviewers mood for games, the collaborative embedding is created using TruncatedSVD to reduce the components of the matrix games x user where the values of this matrix are the user ratings. The table shape will be Number_of_Games x Number_of_Users. After loading this matrix, it will be converted into a sparse matrix and feed into a TruncatedSVD algorithm to compute the new Collaborative Embedding matrix that will have shape Number_of_Games x N, where N is the number of components. In our case, I chose 8 due to better results on source/scripts/test_collaborative_embedding.ipynb, where other number of components would retrieve games that didn't look alike at all. 

To generate the collaborative embedding dict (maps game title to the N component), run:

```bash
python source/scripts/generate_collaborative_embeding.py
```

### Game Description Embedding

After generating the review collaborative embeddings, the game description text embeddings were generated with:

```bash
python source/scripts/generate_game_description_embedding.py
```

This way we can compare two games by their description. 

### Game Title Embedding

The same is done for the game title embeddings generated with:

```bash
python source/scripts/generate_game_title_embedding.py
```

This way we can compare two games by their title.

After that, you can test all embeddings using the source/scripts/test_embeddings.ipynb notebook.

### Game Tags Embedding (genres + keywords)

Multi-hot encoding of the union of genres and keywords per game,
reduced to N components via TruncatedSVD. Captures co-occurrence
patterns between tags. N=32 chosen based on sanity tests in
test_embeddings.ipynb.

```bash
python source/scripts/generate_game_tags_embedding.py
```

### Combined Embedding Table

Concatenate the four per-game embedding pickles plus eight scaled scalar columns into a single matrix `E` of shape `(N, 1584)`, L2-normalizing each embedding block independently and min-max-scaling the scalars. Outputs `data/game_embeddings_matrix.npy`, `data/game_embeddings_index.pkl` (with the filter columns the app uses), and `data/embedding_scalers.pkl`.

```bash
python source/scripts/build_combined_embeddings.py
```

Then sanity-check by opening `source/scripts/test_combined_embeddings.ipynb` and eyeballing the top-10 closest games for a handful of query titles before moving on.

### Train Offline-RL Policy (IQL)

Build the offline-RL MDP dataset from `data/reviews.csv` + the combined embedding and train a continuous-action IQL policy via `d3rlpy`. State = running average of past action vectors; action = game embedding; reward = per-user z-scored review score (rescaled to `[-1, 1]`); terminal = last review per user. (IQL is used in place of CQL, which has a numerical pathology with our 1584-d action space — see `PLAN.md` for the rationale.)

```bash
python source/scripts/build_mdp_dataset.py
python source/scripts/train_iql.py
```

Saves a TorchScript policy to `data/policy.pt` (input dim 1584, output dim 1584).

Then sanity-check the trained policy by opening `source/scripts/test_policy.ipynb` and eyeballing the top-10 recommendations from a cold-start state plus a handful of user profiles (RPG fan, FPS fan, etc.) before moving on.

### Run the HuggingFace App

A Gradio app at the repo root wires everything together: cold-start questionnaire → user-defined filters → top-5 game recommendations with IGDB cover art.

```bash
python app.py
```

Open `http://localhost:7860`. The app has three ranking modes (selectable in the UI): **Profile + RL** (default — keep the games closest to your history, policy reranks them), **RL only** (policy ranks everything), and **Profile only** (pure content-based, no policy).

### Deploying to HuggingFace Spaces

1. The YAML front-matter at the top of this README configures the Space (`sdk: gradio`, `app_file: app.py`, `python_version: "3.11"`).
2. **Upload the four runtime artifacts** from `data/` — `policy.pt`, `game_embeddings_matrix.npy` (~165 MB), `game_embeddings_index.pkl`, `embedding_scalers.pkl`. They are **gitignored** (so the 165 MB matrix never lands in normal git history), so on the Space either track them with `git lfs` (`git lfs track "*.npy"`) or upload them via the Space's web UI / `huggingface_hub`. The training-only inputs (`reviews.csv`, `games.csv`, the per-embedding pickles) are **not** needed at runtime.
3. **Set IGDB secrets** in the Space settings: `IGDB_CLIENT_ID`, `IGDB_CLIENT_SECRET` (the live cover-art toggle degrades gracefully to placeholders if absent). `RAWG_API_KEY` / `GAMESPOT_API_KEY` / `METACRITIC_API_KEY` are only needed for the data-collection scripts, not the app.
4. `requirements.txt` covers the whole pipeline; the Space will also install the training-only `d3rlpy` (its `torch` dependency *is* used at inference, via the TorchScript policy). For a leaner image you can drop `d3rlpy`/`sentence-transformers` from a Space-only requirements file — the app itself only needs `gradio`, `torch`, `pandas`, `numpy`, `scikit-learn`, `thefuzz`, `python-dotenv`, `Pillow`, `requests`.

## Status & known limitations

The pipeline runs end-to-end and the app produces sensible recommendations for users with a play history. The honest caveat: the trained policy's *action vector* tends to collapse toward the embedding centroid (a known failure mode of advantage-weighted regression on a high-dimensional continuous action with weakly-contrasting rewards), so recommendation quality currently leans on the candidate generator's profile rerank (`profile_prefilter=True`, keep the 30 games closest to the play history, policy reranks those) rather than the policy alone. Per-user z-scored rewards were introduced to give the policy more contrast to learn from. Cold-start (no history) is the regime where the policy's weakness is most visible. `PLAN.md` → "Known limitations & follow-up strategies" tracks what's been tried (more training steps; reward sharpening) and the remaining levers (lower-dim action target; discrete action-ID ranking over the candidate set).

See `PLAN.md` for the full strategy and `ideia.txt` for the long-form design log.