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

In game-advisor folder run:

```bash
python source/scripts/create_user_review_dataset.py
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

Build the offline-RL MDP dataset from `data/reviews.csv` + the combined embedding and train a continuous-action IQL policy via `d3rlpy`. State = running average of past action vectors; action = game embedding; reward = `(score - 5) / 5`; terminal = last review per user. (IQL is used in place of CQL, which has a numerical pathology with our 1584-d action space — see `PLAN.md` for the rationale.)

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

Open `http://localhost:7860`. To deploy as a HuggingFace Space, push the repo to a Space with `sdk: gradio`; ship the four artifacts in `data/` (`policy.pt`, `game_embeddings_matrix.npy`, `game_embeddings_index.pkl`, `embedding_scalers.pkl`) and put the IGDB credentials in the Space's secrets.

See `PLAN.md` for the full strategy and `ideia.txt` for the long-form design log.