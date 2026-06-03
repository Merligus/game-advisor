# Strategy: ship game-advisor to a HuggingFace Space

This file captures the path from the current state of the repo to a deployed offline-RL game recommender on HuggingFace Spaces. `ideia.txt` remains the long-form design log (in Portuguese); this file is the executable strategy.

## Load-bearing decisions

- **Continuous action space.** The offline-RL policy (IQL — see Stage 3) emits a game-embedding vector; we nearest-neighbor it inside a filtered candidate set to produce top-K recommendations. The alternative — discrete actions over ~26k game IDs — would need a 26k-way Q-head. *This decision is the suspected root cause of the centroid-collapse problem (see "Known limitations & follow-up strategies"); discrete action-ID over the candidate set is now an open follow-up.*
- **Full 1584-d combined embedding.** Per-game vector `E` = concat(title 768, description 768, tags 32, collab 8, scalars 8). No PCA reduction step. Change the concat order or per-block normalization and you invalidate the trained policy.

## Stages

### Stage 1 — Combined embedding table

`source/scripts/build_combined_embeddings.py` reads the four existing pickles in `data/` plus the eight scalar columns of `games.csv` (`metacritic_rating, igdb_rating, rawg_rating, hltb_rating, user_rating, main_story, main_extra, completionist`), L2-normalizes each embedding block independently, min-max scales the scalars to `[0,1]` and divides them by `sqrt(8)` so cosine isn't dominated by the scalars, imputes missing blocks with the block mean over present games, drops games still missing the title block, and writes:

- `data/game_embeddings_matrix.npy` — `float32`, shape `(N, 1584)`
- `data/game_embeddings_index.pkl` — `name, row_idx` + filter columns (`release_year, platforms, language_supports, main_story, main_extra, completionist`). `release_year` is clamped to NaN beyond `MAX_PLAUSIBLE_YEAR` (2035) to drop IGDB far-future TBA placeholders (clustered at 2097-2099); they become "unknown" rather than bogus future releases. Rebuilding the index this way leaves `E` byte-identical (year is index-only metadata, not a model feature), so `policy.pt` stays valid.
- `data/embedding_scalers.pkl` — scaler params + per-block imputation means, so inference normalizes new inputs identically

### Stage 1.5 — Sanity-check the combined embedding (notebook)

`source/scripts/test_combined_embeddings.ipynb` mirrors `test_embeddings.ipynb` but loads the matrix+pickle pair and runs `find_closest_games` with `top_n=10` against a handful of pre-filled query games covering distinct genres (RPG, shooter, strategy, indie). **Gate before Stage 2**: advance only once the top-10 lists look reasonable to a human reviewer.

### Stage 2 — MDP dataset

`source/scripts/build_mdp_dataset.py` builds the continuous-action MDP from `data/reviews.csv` + the combined embedding. For each user (sorted by review date), iterate reviews and emit `(state, action, reward, terminal)`: state = running average of past action vectors (zero vector at t=0, dim 1584); action = E-row of the rated game; reward = **per-user z-scored review score** — `(score − user_mean) / user_std`, clipped to `[-3, 3]` and divided by 3 → `[-1, 1]` (users with <2 reviews or zero rating variance fall back to global mean/std centering); terminal = last review per user. The reward was originally the global `(score − 5) / 5`, but that distribution was mean **+0.41 / only ~17% negative** — too flat for IQL's advantage estimate to discriminate states, which collapsed the actor (see "Known limitations & follow-up strategies"). Z-scoring centers each user on their own taste, yielding mean **≈0 / ~42% negative** — real contrast. Drop reviews with null author/game_name and reviews whose `game_name` isn't in the embedding index. Saves `data/mdp_dataset.npz` (compressed) containing `observations` (N, 1584) `float32`, `action_row_idx` (N,) `int32`, `rewards` (N,) `float32`, `terminals` (N,) `float32`, plus `data/mdp_meta.json`. We store action *indices* into `E` rather than materialized action vectors so the file stays ~1 GB and auto-syncs if `E` is rebuilt; Stage 3 reconstructs `actions = E[action_row_idx]` before passing to `d3rlpy.dataset.MDPDataset`. Stays dep-free of `d3rlpy`.

### Stage 3 — Offline-RL training (IQL)

`source/scripts/train_iql.py` instantiates `d3rlpy.algos.IQLConfig(action_scaler=MinMaxActionScaler(), gamma=0.2, batch_size=256, actor_learning_rate=3e-4, critic_learning_rate=3e-4)` (other IQL knobs — `expectile=0.7`, `weight_temp=3.0`, `max_weight=100` — left at d3rlpy defaults). First end-to-end pass: `n_steps=50_000`; bump to `200_000` once the pipeline is green. Holds out the last 10% of each user's reviews and off-policy-evaluates the trained policy with `d3rlpy.ope.FQE` (`InitialStateValueEstimationEvaluator`). Saves TorchScript policy to `data/policy.pt` (input dim 1584, output dim 1584) and a training summary to `data/training_log.json`.

> **Why IQL and not CQL (the plan originally called for CQL):** d3rlpy's CQL implementation computes `math.log(0.5**action_size)` to get the log-density of uniform random actions over `[-1, 1]^d`. With `action_dim=1584`, `0.5**1584` underflows to `0.0` in float64 and the call raises `ValueError: math domain error`. The stable form `1584 * math.log(0.5) ≈ -1098` is mathematically correct, but even patched, that constant +1098 offset on the random-action importance term swamps the data Q-value signal inside CQL's conservative loss — the offline-safety regularization becomes functionally degenerate at this action dim. IQL provides the same offline-safety guarantee (expectile-clipped value learning + AWR-style policy extraction) without that pathology and is the modern offline-RL default for continuous high-dim actions. Same algorithmic role; cleaner scaling.

> **Training-steps note:** 50K is the diagnostic run (behaviour is determined by ~50K — see the follow-up section for the 50K-vs-200K A/B); bump to 200K only once a change is shown to move behaviour. FQE uses the holdout solely as a sanity scalar — its absolute value isn't comparable across reward definitions (the z-score reward rescaled the return), so read it as "finite and positive", not as a quality metric.

### Stage 3.5 — Sanity-check the trained policy (notebook)

`source/scripts/test_policy.ipynb` mirrors the Stage 1.5 pattern but for the trained policy. Loads `data/policy.pt`, `data/game_embeddings_matrix.npy`, and `data/game_embeddings_index.pkl`. Defines two helpers: `build_state(played_games)` (mean of E-vectors over recognized games; zeros for cold-start) and `top_k_recommendations(state, top_n=10)` (runs the policy on `state`, returns top-N games by cosine to the predicted action vector, filtering out already-played). Pre-filled cells exercise distinct profiles — cold start, RPG fan (Witcher 3 + Dark Souls III + Skyrim), FPS fan (DOOM Eternal + Halo Infinite + Titanfall 2), strategy fan (Civ VI + Total War: WARHAMMER), indie fan (Hollow Knight + Celeste + Stardew Valley) — plus an ad-hoc empty cell. **Gate before Stage 4**: advance only when the policy's recommendations diverge appropriately across profiles and its cold-start picks look plausible.

### Stage 4 — Candidate generator

`source/recommender/candidate_generator.py::candidates(filters, played_games=None, k=500)` filters `game_embeddings_index.pkl` by year range, platform match, and language substring, then optionally reranks by cosine against the played-games mean profile and trims to `k` (`k=None` disables the cap). Uses `sklearn.metrics.pairwise.cosine_similarity` (already in `requirements.txt`).

**Filter semantics — lenient on missing data**: a game is excluded only when its metadata is *present and contradicts* the filter. This is load-bearing because `language_supports` covers only ~6% of the catalog and `release_year` ~83%; strict matching on missing fields would empty the result set. Platform matching is case-insensitive **one-directional** substring (requested label ⊂ game label) so `"PC"` matches `"PC (Microsoft Windows)"` — but a `"PlayStation 5"` request does not match a game labeled only `"PlayStation"` (the bidirectional version over-matched PS1 titles into PS5 filters; caught by the Stage 5 compliance check).

App modules share `source/recommender/artifacts.py` — cached (`lru_cache`) loaders for the embedding matrix, normalized matrix, index frame, `name → row` map, and the CPU-loaded TorchScript policy. Paths resolve relative to the file so imports are cwd-independent; the 165 MB matrix is read once per process. (This shared loader wasn't in the original plan but keeps Stages 4-6 from each re-reading the artifacts.)

### Stage 5 — Inference

`source/recommender/inference.py::recommend(state, filters, played_games, top_n=5, profile_prefilter=True, candidate_k=30)` calls the candidate generator, runs the TorchScript policy on `state`, scores candidates by cosine against the predicted action vector, filters out games already in `played_games`, and returns dicts `{name, score, release_year, platforms, cover_url, description}` joined from the pickle index + cached `games.csv` (`artifacts.games_metadata()`). Optional live IGDB cover-art refresh via `source/APIs/igdb_api.py::IGDB.search`. **Default is profile rerank** (`profile_prefilter=True`, `candidate_k=30`): the candidate generator keeps the 30 games closest to the play history and the policy reranks those — anchoring recommendations to the history and masking the policy's current undertraining. Cold start (no history) falls back to filter-only with `k=None` so the policy ranks the whole filtered set (also dodges the alphabetical-truncation artifact). `profile_prefilter=False` always lets the policy rank the full filtered set — the "trust the policy" mode, preferable once the policy is well trained. The two modes are why "Battlefield Hardline: Getaway" appears under `profile_prefilter=False` but not under the default: it passes the PC/2010+ filter and the policy's action points near it, but it ranks 593rd in raw similarity to an RPG history, far outside the kept 30.

### Stage 6 — Gradio app on HuggingFace

`app.py` at the repo root is the Space entry point — a single-page `gr.Blocks`:

- **Play history** — one searchable `gr.Dropdown(multiselect=True, filterable=True, choices=all_game_names)`: type a title, the closest catalog matches appear, pick the ones you've played. (`state_builder.cold_start_state` still fuzzy-resolves names and keeps the per-input resolution available, but the UI is a single search box rather than a dropdown + free-text pair.) Submit → builds the state via `source/recommender/state_builder.py::cold_start_state`.
- **Filters** — `gr.Slider` × 2 (year range), `gr.CheckboxGroup` (top-20 platforms from the index), `gr.Dropdown` (language).
- **Recommendations** — `gr.Gallery` of 5 covers + a `gr.Markdown` details panel per game. A "Mark as played → refine" multiselect folds picks back into the history and re-runs.
- **Cover art** — `cover_url` from `games.csv` when present, else a PIL placeholder; a default-on "Fetch cover art live from IGDB" checkbox fills missing covers via `source/APIs/igdb_api.py` (creds from `.env` / Space secrets), upgrading IGDB `t_thumb` URLs to `t_cover_big`.

`requirements.txt` gains `gradio==6.15.1` (and `d3rlpy==2.8.1`); torch stays host-installed (numpy 2.x is fine — no pin needed). Shippable artifacts the Space needs at runtime: `data/policy.pt`, `data/game_embeddings_matrix.npy`, `data/game_embeddings_index.pkl`, `data/embedding_scalers.pkl`, plus `source/recommender/` and `source/APIs/` (IGDB credentials go in the Space's secrets).

### Stage 7 — Docs

This file (`PLAN.md`) plus the extensions to `README.md` (one section per new stage with the single command to invoke it) and `CLAUDE.md` (pipeline steps 8–12, the combined-embedding contract, and the inference subsection).

## Known limitations & follow-up strategies

### Problem: the continuous-action policy collapses to the embedding centroid

**Symptom (measured).** Under `profile_prefilter=False` ("let the policy rank the whole filtered set"), the policy returns nearly the same games regardless of the user state, and the predicted-action cosines to candidate games are all bunched around ~0.74 — i.e. the actor outputs roughly the embedding **centroid** rather than a state-specific direction. The cold-start top-5 is a fixed "default cluster" (Revenge of Arcade, BattleZone, Rayman 2, …).

**Why.** Three compounding causes: (1) a 1584-d *continuous* action extracted by IQL's advantage-weighted regression converges toward a near-constant conditional mean; (2) the original reward `(score−5)/5` was mostly positive (~0.41 mean, ~17% negative), so advantages barely contrasted states and the AWR weights were near-uniform; (3) the cold-start zero-state carries no signal. It is **structural, not under-training** — losses kept improving 50K→200K while recommendations stayed identical.

### What's been tried

- **More steps (50K → 200K).** Losses improved (critic 0.45→0.11) but cold-start picks were *identical* and FQE flat (1.484→1.450). Confirms collapse is not an under-training artifact. Behaviour is locked in by ~50K, so 50K is the standard diagnostic run.
- **Sharper rewards — per-user z-score (option 1, applied in Stage 2).** Replaces the flat global reward (mean +0.41, 17% neg) with a per-user-centered one (mean ≈0, ~42% neg) so the advantage estimate has contrast. *Evaluated via a 50K A/B against the original collapsed 50K run — see the latest `data/training_log.json` and the `test_policy.ipynb` divergence diagnostic for the outcome.*

### Remaining levers (not yet tried)

- **Lower-dim action target.** Predict into a compressed action space (e.g. PCA E → 64–128d) and map back; less room for centroid collapse than regressing 1584-d directly.
- **Discrete action-ID over the candidate set.** Reframe as ranking: have the policy score each of the ≤K candidate games (`Q(state, game)→scalar`) and argmax, instead of regressing a vector. This is how most production recommenders work and sidesteps the continuous-mean collapse, but it's a re-architecture of Stages 2/3/5 and fights d3rlpy's fixed-action-space assumption (would likely need a custom scorer rather than an off-the-shelf discrete algo). Revisits the Stage-0 continuous-vs-discrete decision now that candidate generation bounds the action set.

### Current mitigation (load-bearing)

The **profile-rerank default** (`profile_prefilter=True`, `candidate_k=30`) means the candidate generator's cosine-to-history — *not* the policy — drives recommendation quality: it keeps the 30 games closest to the play history and the policy only reranks those. This is why warm-start recommendations look good despite the collapse. **Do not remove the profile rerank assuming the policy will rank well on its own** until one of the levers above demonstrably fixes the collapse; cold-start (no history) is the regime where the policy's weakness is still visible.

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
