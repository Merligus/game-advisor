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

`source/scripts/build_mdp_dataset.py` builds the continuous-action MDP from `data/reviews.csv` + the combined embedding. For each user (sorted by review date), iterate reviews and emit `(state, action, reward, terminal)`: state = running average of past action vectors (zero vector at t=0, dim 1584); action = E-row of the rated game; reward = `(score - 5) / 5` (already in `[-1, 1]` since scores are `[0, 10]`); terminal = last review per user. Drop reviews with null author/game_name and reviews whose `game_name` isn't in the embedding index. Saves `data/mdp_dataset.npz` (compressed) containing `observations` (N, 1584) `float32`, `action_row_idx` (N,) `int32`, `rewards` (N,) `float32`, `terminals` (N,) `float32`, plus `data/mdp_meta.json`. We store action *indices* into `E` rather than materialized action vectors so the file stays ~1 GB and auto-syncs if `E` is rebuilt; Stage 3 reconstructs `actions = E[action_row_idx]` before passing to `d3rlpy.dataset.MDPDataset`. Stays dep-free of `d3rlpy`.

### Stage 3 — Offline-RL training (IQL)

`source/scripts/train_iql.py` instantiates `d3rlpy.algos.IQLConfig(action_scaler=MinMaxActionScaler(), gamma=0.2, batch_size=256, actor_learning_rate=3e-4, critic_learning_rate=3e-4)` (other IQL knobs — `expectile=0.7`, `weight_temp=3.0`, `max_weight=100` — left at d3rlpy defaults). First end-to-end pass: `n_steps=50_000`; bump to `200_000` once the pipeline is green. Holds out the last 10% of each user's reviews and off-policy-evaluates the trained policy with `d3rlpy.ope.FQE` (`InitialStateValueEstimationEvaluator`). Saves TorchScript policy to `data/policy.pt` (input dim 1584, output dim 1584) and a training summary to `data/training_log.json`.

> **Why IQL and not CQL (the plan originally called for CQL):** d3rlpy's CQL implementation computes `math.log(0.5**action_size)` to get the log-density of uniform random actions over `[-1, 1]^d`. With `action_dim=1584`, `0.5**1584` underflows to `0.0` in float64 and the call raises `ValueError: math domain error`. The stable form `1584 * math.log(0.5) ≈ -1098` is mathematically correct, but even patched, that constant +1098 offset on the random-action importance term swamps the data Q-value signal inside CQL's conservative loss — the offline-safety regularization becomes functionally degenerate at this action dim. IQL provides the same offline-safety guarantee (expectile-clipped value learning + AWR-style policy extraction) without that pathology and is the modern offline-RL default for continuous high-dim actions. Same algorithmic role; cleaner scaling.

### Stage 3.5 — Sanity-check the trained policy (notebook)

`source/scripts/test_policy.ipynb` mirrors the Stage 1.5 pattern but for the trained policy. Loads `data/policy.pt`, `data/game_embeddings_matrix.npy`, and `data/game_embeddings_index.pkl`. Defines two helpers: `build_state(played_games)` (mean of E-vectors over recognized games; zeros for cold-start) and `top_k_recommendations(state, top_n=10)` (runs the policy on `state`, returns top-N games by cosine to the predicted action vector, filtering out already-played). Pre-filled cells exercise distinct profiles — cold start, RPG fan (Witcher 3 + Dark Souls III + Skyrim), FPS fan (DOOM Eternal + Halo Infinite + Titanfall 2), strategy fan (Civ VI + Total War: WARHAMMER), indie fan (Hollow Knight + Celeste + Stardew Valley) — plus an ad-hoc empty cell. **Gate before Stage 4**: advance only when the policy's recommendations diverge appropriately across profiles and its cold-start picks look plausible.

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
python source/scripts/train_iql.py
# open source/scripts/test_policy.ipynb, eyeball recs across user profiles, then continue
python app.py   # then open http://localhost:7860
```

Per-stage gates: Stage 1 prints `(N, 1584)`; Stage 1.5 user eyeballs top-10 neighbors per query; Stage 2 prints reward histogram + matched terminal count; Stage 3 logs a finite FQE scalar and writes `data/policy.pt`; Stage 3.5 user eyeballs policy recs across cold-start + 4 user profiles; Stage 6 boots Gradio and renders 5 covers for a smoke-test query.
