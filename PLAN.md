# Strategy: ship game-advisor to a HuggingFace Space

This file captures the path from the current state of the repo to a deployed offline-RL game recommender on HuggingFace Spaces. `ideia.txt` remains the long-form design log (in Portuguese); this file is the executable strategy.

## Load-bearing decisions

- **Continuous action space.** The CQL policy emits a game-embedding vector; we nearest-neighbor it inside a filtered candidate set to produce top-K recommendations. Matches `source/snippets/train_step.py`. The alternative — discrete actions over ~20k game IDs — would require a 20k-way Q-head and doesn't compose with candidate generation.
- **Full 1584-d combined embedding.** Per-game vector `E` = concat(title 768, description 768, tags 32, collab 8, scalars 8). No PCA reduction step. Change the concat order or per-block normalization and you invalidate the trained policy.

## Stages

### Stage 1 — Combined embedding table

`source/scripts/build_combined_embeddings.py` reads the four existing pickles in `data/` plus the eight scalar columns of `games.csv` (`metacritic_rating, igdb_rating, rawg_rating, hltb_rating, user_rating, main_story, main_extra, completionist`), L2-normalizes each embedding block independently, min-max scales the scalars to `[0,1]` and divides them by `sqrt(8)` so cosine isn't dominated by the scalars, imputes missing blocks with the block mean over present games, drops games still missing the title block, and writes:

- `data/game_embeddings_matrix.npy` — `float32`, shape `(N, 1584)`
- `data/game_embeddings_index.pkl` — `name, row_idx` + filter columns (`release_year, platforms, language_supports, main_story, main_extra, completionist`)
- `data/embedding_scalers.pkl` — scaler params + per-block imputation means, so inference normalizes new inputs identically

### Stage 1.5 — Sanity-check the combined embedding (notebook)

`source/scripts/test_combined_embeddings.ipynb` mirrors `test_embeddings.ipynb` but loads the matrix+pickle pair and runs `find_closest_games` with `top_n=10` against a handful of pre-filled query games covering distinct genres (RPG, shooter, strategy, indie). **Gate before Stage 2**: advance only once the top-10 lists look reasonable to a human reviewer.

### Stage 2 — MDP dataset

`source/scripts/build_mdp_dataset.py` builds the continuous-action MDP from `data/reviews.csv` + the combined embedding. For each user (sorted by review date), iterate reviews and emit `(state, action, reward, terminal)`: state = running average of past action vectors (zero vector at t=0, dim 1584); action = E-row of the rated game; reward = `(score - 5) / 5` clipped to `[-1, 1]`; terminal = last review per user. Drop reviews whose `game_name` isn't in the embedding index. Saves `data/mdp_dataset.h5` via `d3rlpy.dataset.MDPDataset.dump`.

### Stage 3 — CQL training

`source/scripts/train_cql.py` instantiates `d3rlpy.algos.CQL(action_scaler="min_max", gamma=0.2, actor_learning_rate=1e-4, critic_learning_rate=3e-4, batch_size=256, use_gpu=<auto>)`. First end-to-end pass: `n_steps=50_000`; bump to `200_000` once the pipeline is green. Holds out the last 10% of each user's reviews for `d3rlpy.ope.FQE`. Saves TorchScript policy to `data/policy.pt` (input dim 1584, output dim 1584) and training scalars to `data/training_log.json`.

### Stage 4 — Candidate generator

`source/app/candidate_generator.py::candidates(filters, played_games=None, k=500)` filters `game_embeddings_index.pkl` by year range, platform intersection, and language substring, then optionally reranks by cosine against the played-games mean profile and trims to `k`. Uses `sklearn.metrics.pairwise.cosine_similarity` (already in `requirements.txt`).

### Stage 5 — Inference

`source/app/inference.py::recommend(state, filters, played_games, top_n=5)` calls the candidate generator, runs the TorchScript policy on `state`, scores candidates by cosine against the predicted action vector, filters out games already in `played_games`, and returns dicts `{name, score, cover_url, description, release}` joined from the pickle index + cached `games.csv`. Optional live IGDB cover-art refresh via `source/APIs/igdb_api.py::IGDB.search`.

### Stage 6 — Gradio app on HuggingFace

`app.py` at the repo root is the Space entry point. Three Gradio tabs:

1. **Cold start** — `gr.Dropdown(multiselect=True, choices=all_game_names)` + free-text fuzzy fallback (`thefuzz.fuzz.ratio`). Submit → builds H0 via `source/app/state_builder.py::cold_start_state(played_games)` (mean of E-vectors; zero vector if none recognized).
2. **Filters** — `gr.Slider` (year range), `gr.CheckboxGroup` (platforms, derived from the index), `gr.Dropdown` (language).
3. **Recommendations** — `gr.Gallery` of 5 covers + collapsible `gr.Markdown` per game; per-item "I played this" button appends to `played_games` and re-runs `recommend`.

`requirements.txt` gains `gradio`, `d3rlpy`, `torch` (CPU wheel index URL for HF Spaces), and pins `numpy<2` if `d3rlpy` requires it. Shippable artifacts the Space needs at runtime: `data/policy.pt`, `data/game_embeddings_matrix.npy`, `data/game_embeddings_index.pkl`, `data/embedding_scalers.pkl`, and `source/APIs/` for live cover art (IGDB credentials go in the Space's secrets).

### Stage 7 — Docs

This file (`PLAN.md`) plus the extensions to `README.md` (one section per new stage with the single command to invoke it) and `CLAUDE.md` (pipeline steps 8–11, the combined-embedding contract, and the inference subsection).

## End-to-end run order

```bash
python source/scripts/build_combined_embeddings.py
# open source/scripts/test_combined_embeddings.ipynb, eyeball top-10s, then continue
python source/scripts/build_mdp_dataset.py
python source/scripts/train_cql.py
python app.py   # then open http://localhost:7860
```

Per-stage gates: Stage 1 prints `(N, 1584)`; Stage 1.5 user eyeballs top-10 neighbors per query; Stage 2 prints reward histogram + matched terminal count; Stage 3 logs a finite FQE scalar and writes `data/policy.pt`; Stage 6 boots Gradio and renders 5 covers for a smoke-test query.
