# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Offline reinforcement-learning game recommender. Pulls game metadata + user ratings from public APIs, builds several game embeddings, and trains an offline-RL policy (IQL via `d3rlpy`; see `PLAN.md` for why IQL was substituted for CQL) for deployment as a HuggingFace app. `ideia.txt` is the working design doc — written in Portuguese, kept up-to-date by the user, and the source of truth for what's done vs. planned. Read it before making non-trivial changes.

## Environment

```bash
conda create -n GameAdvisor python=3.11 -y
conda activate GameAdvisor
pip install -r requirements.txt
```

A `.env` at the project root is required for the data-fetching scripts:

```
GAMESPOT_API_KEY=
RAWG_API_KEY=
IGDB_CLIENT_ID=
IGDB_CLIENT_SECRET=
METACRITIC_API_KEY=   # used by source/APIs/metacritic_api.py and metacritic_create_user_review_dataset.py
```

`requirements.txt` does **not** install PyTorch — embedding scripts call `SentenceTransformer(..., device="cuda")` and the IQL training script uses `torch.cuda`, so a CUDA-capable PyTorch must be installed separately (or the `device` flag changed). The repo was trained on `torch==2.11.0+cu130` with an RTX 3050 6GB. `d3rlpy==2.8.1` is now pinned in `requirements.txt`; the MDP/training code in `source/snippets/` remains reference material — runtime training happens via `source/scripts/train_iql.py`.

## Running the pipeline

All scripts are designed to be invoked **from the project root** — they use relative paths like `./data/...`. Run order matters because each stage consumes the previous stage's CSV:

1. `python source/scripts/gamespot_create_user_review_dataset.py` → `data/gamespot_reviews.csv`
2. `python source/scripts/metacritic_create_user_review_dataset.py` → `data/metacritic_reviews.csv` (seeds itself from gamespot's unique games)
3. `python source/scripts/merge_reviews.py` → `data/reviews.csv` (renames `authors`→`author`, `publish_date`→`date`, dedupes, concats)
4. `python source/scripts/create_game_dataset.py` → `data/games.csv` (long-running, rate-limited; see below)
5. Notebook `source/scripts/update_table_version_2.ipynb` — drops empty/duplicate `name` rows and writes a cleaned `games_2.csv`
6. Embedding generators (each writes a `{game_name: vector}` dict into `data/*.pkl`):
   - `generate_collaborative_embeding.py` — TruncatedSVD on the user×game rating matrix, N=8
   - `generate_game_description_embedding.py` — `all-mpnet-base-v2` over `description`
   - `generate_game_title_embedding.py` — `all-mpnet-base-v2` over `name`
   - `generate_game_tags_embedding.py` — multi-hot of `genres ∪ keywords` → TruncatedSVD, N=32
   - `generate_game_tags_semantic_embedding.py` — tested alternative (SBERT over concatenated tag string); **not** used downstream per `ideia.txt`
7. Sanity-check embeddings via `test_embeddings.ipynb` (cosine top-k) and `test_embeddings_compare.ipynb` (side-by-side of two embedding files).
8. `python source/scripts/build_combined_embeddings.py` — concatenate the four embedding pickles + scaled scalars into `data/game_embeddings_matrix.npy` (shape `(N, 1584)`), `data/game_embeddings_index.pkl`, and `data/embedding_scalers.pkl`. L2-norm per block; scalars min-maxed and divided by `sqrt(8)`.
9. Sanity-check the combined embedding via `source/scripts/test_combined_embeddings.ipynb` — top-10 cosine neighbors per query game. **Human gate before step 10**; the next stage encodes the embedding contract into a trained policy.
10. `python source/scripts/build_mdp_dataset.py` — build the continuous-action MDP from `reviews.csv` + the combined embedding; saves `data/mdp_dataset.npz` (observations + `action_row_idx` indices into `E` + rewards + terminals; Stage 3 reconstructs actions on load).
11. `python source/scripts/train_iql.py` — train `d3rlpy.algos.IQL` (gamma=0.2, `MinMaxActionScaler`), hold out the last 10% of each user's reviews for FQE, save TorchScript policy to `data/policy.pt` (input/output dim 1584). IQL replaces CQL here because CQL's conservative loss underflows at action_dim=1584; see `PLAN.md` Stage 3 for the rationale.
12. Sanity-check the trained policy via `source/scripts/test_policy.ipynb` — top-10 recommendations from a cold-start state plus four user profiles (RPG/FPS/strategy/indie). **Human gate before Stage 4**; the policy's behavior here is what the Gradio app will surface to end users.

See `PLAN.md` for the strategy that introduced steps 8–12 and the HuggingFace deployment that follows.

For the long-running step 4, `source/scripts/call_script_periodically.sh` is a `fish` loop that re-invokes `create_game_dataset.py` every 10 min — it's used because the script is resumable and rate limits force frequent restarts.

There are no tests, no linter config, and no build step.

## Architecture

### `source/APIs/` — uniform API wrappers
Each module exposes a class with a `.search(game_name, max_n) -> list[<APIType>]` method. The returned dataclasses all extend `GameType` (`api_types.py`), so downstream code can treat them uniformly while still accessing API-specific extras (e.g. `IGDBType.themes`, `MetacriticType.slug`). When adding a new API source, mirror this pattern — define an `<X>Type(GameType)` dataclass and a class with the same `.search` signature.

RAWG/IGDB/Metacritic share **`http_client.py`** (`requests.Session` reuse, 20s timeout on every call, retry with backoff on 429/5xx honoring `Retry-After`); network failure after retries raises, and callers (`enrichment.py`, `create_game_dataset.py`) catch per source. Gamespot has its own fetcher stack (curl_cffi impersonation → Playwright fallback, with Cloudflare-challenge detection); HLTB goes through `howlongtobeatpy` (**pin ≥ 1.0.22** — 1.0.19 silently returned zero results, which is why old repairs show 0.0 time-to-beat).

Live-API status (as used by `recommender/enrichment.py` + `scripts/update_games.py`): **RAWG** is the reliable source for descriptions + covers; `search` uses `search_precise`, and it also exposes `get_by_id(id)` and `list_recent(start, end, max_results, min_added)` (popularity-ordered discovery of recent releases — the updater's "new games" source). **IGDB** gives portrait box-art covers + summaries + developers/publishers (via `involved_companies`) and refreshes an expired token on 401; its `search` clause takes the **raw** string with `"`/`\` escaped — *not* URL-encoded — and its search is loose (returns bundles/spinoffs) so callers must name-match-guard its hits. **Metacritic** locates critic/user score components by identity scan (falling back to the historical `components[6]/[8]` indices); carries scores + devs/pubs but no cover/description. **Gamespot** works live through the WordPress API behind the challenge-aware fetchers. **HLTB** carries time-to-beat + a review score; treat empty results as degradation (its `search` never raises).

### `source/scripts/create_game_dataset.py` — the merge layer
This is the only place where the five APIs are reconciled into a single per-game row. Key behaviors that future edits must preserve:

- **Resumable**: on startup, reads existing `data/games.csv` and seeds `already_processed` with the `real_name` column so reruns skip done games. The CSV is appended (not rewritten) every 10 successful games — never replace this with an in-memory accumulator unless you also handle resume.
- **Fuzzy match acceptance rule** (`nameRatio` via `thefuzz.fuzz.ratio`, normalized to 0–1):
  - Gamespot is the anchor and uses `minRatio=0.65` alone (its release date becomes `game_release`, which everything else is compared against).
  - For RAWG/IGDB/HLTB/Metacritic: accept if `ratio > goodRatio (0.9)`, OR `ratio > minRatio (0.65)` AND release year within 1 of the Gamespot anchor (`compareRelease`). Otherwise the result is discarded and replaced with the empty dataclass instance.
- **Field-merge priority** (in `GameType(...)` construction): for scalars the first non-empty wins (`getFirstString`/`getFirstFloat`); for lists, either first-non-empty (`getFirstList`) or full union (`getUnion`). The ordering of the source list in each call encodes which API is trusted most for that field — change orderings deliberately.
- **Rate-limit handling**: catches exceptions with "rate limit"/"429"/"420"/"too many requests"/"quota" in the message and sleeps `rateLimitBackoff=60`s up to `maxRateLimitRetries=3` before bailing. Generic errors get `maxRetries=3` quick retries before skipping the game.

### Embeddings — output contract
Every `*_embedding.pkl` in `data/` is a `dict[str, np.ndarray]` keyed by the game's `name` (not `real_name`). The test notebooks rely on this shape; keep it consistent across new generators. Games whose tag set / description / SVD row would be a zero-vector are dropped (see `generate_game_tags_embedding.py` — zero vectors break cosine similarity).

### Combined embedding `E` (dim 1584)
Stage 1 of the deployment pipeline (`source/scripts/build_combined_embeddings.py`; see `PLAN.md`) concatenates: title SBERT (768) + description SBERT (768) + tags SVD (32) + collab SVD (8) + 8 scaled scalars. Each of the four embedding blocks is L2-normalized independently; scalars are min-maxed to `[0,1]` then divided by `sqrt(8)` so they don't dominate cosine. Missing blocks are imputed with the per-block mean over present games; games still missing the title block are dropped. The matrix lives at `data/game_embeddings_matrix.npy` and is the **only** representation that downstream stages (MDP build, IQL training, inference) read — change the concat order, the per-block normalization, or the scalar scaling and you invalidate `data/policy.pt`.

### Inference layer — `source/recommender/`
The HuggingFace app composes these modules:
- `artifacts.py` — `lru_cache`d loaders for the embedding matrix, normalized matrix, index frame, `name → row` map, and the CPU-loaded TorchScript policy. Paths resolve relative to the file (`parents[2]/data`), so imports are cwd-independent and the 165 MB matrix is read once per process. The other app modules import from here. **Built.**
- `candidate_generator.py::candidates(filters, played_games=None, k=500)` — filters `game_embeddings_index.pkl` (year range, platform match, language substring) + optional cosine rerank against the user's played-games mean profile; returns ≤K row indices into the embedding matrix (`k=None` disables the cap). **Filters are lenient on missing data** (a game is excluded only when its metadata is present and contradicts the filter) because language coverage is ~6% and year ~83% — strict matching would empty the catalog. Platform matching is case-insensitive **one-directional** substring (requested label ⊂ game label): `"PC"` matches `"PC (Microsoft Windows)"`, but a `"PlayStation 5"` request does **not** match a game labeled only `"PlayStation"` (the reverse direction over-matched PS1 games into PS5 filters). **Built.**
- `inference.py::recommend(state, filters, played_games, top_n=5, mode="profile_rl", candidate_k=30, enrich=False)` — the per-request hot path: candidate set → rank → drop played → enrich with year/cover/description from `games.csv`, and when `enrich=True` fill any missing cover/description via `enrichment.live_enrich`. Three `MODES`, exposed as a UI radio: **`profile_rl`** (default — keep the `candidate_k=30` history-closest games, policy reranks them; masks the centroid collapse), **`rl_only`** (policy ranks the whole filtered set — raw policy, currently weak), **`profile_only`** (pure cosine-to-history, no policy — content-based baseline). `profile_prefilter` bool kept as a legacy alias (True→profile_rl, False→rl_only). **Built.**
- `enrichment.py::live_enrich(name)` — `games.csv` is sparse (empty cover/description) for the edition-suffixed/niche titles the policy tends to surface. This queries RAWG + IGDB live and merges the best cover + description, **guarded by a fuzzy name-match** (`thefuzz`, ratio ≥ 0.75) so loose hits are rejected. Cover preference: IGDB box art (when accepted) → RAWG `background_image`; description: RAWG (richest) → IGDB summary. `lru_cache`d. Gamespot is currently bot-walled (HTML challenge, not XML) so it's not used live; Metacritic/HLTB carry no cover/description. **Built.**
- `state_builder.py::cold_start_state(played_games)` — mean of E-vectors for the provided games (zero vector if none recognized); resolves user-typed titles to canonical `name` via `thefuzz.fuzz.ratio`. **Built.**

The Gradio entry point is `app.py` at the **repo root** (HF Spaces requires that filename). It prepends `source/` to `sys.path` and imports the `recommender` package. **Note the package is `source/recommender/`, not `source/app/`** — it was renamed because a `source/app/` package collides with the `app.py` entry point (both import as `app`).

### Catalog repair — audit + targeted top-up (`source/scripts/audit_catalog.py`, `add_games.py`)
The batch merge lost famous games to two fuzzy-match bugs (ratio > 0.9 bypassed the release-year check, so "God of War II" merged into the God of War III row; name-identical pairs like God of War 2005/2018 were undecidable). The repair path:

- **Hardened guard** (`recommender/enrichment.py`): candidates whose *sequel-number sets* differ are vetoed at any ratio (roman numerals normalized to arabic, parenthetical years treated as metadata, edition suffixes stripped); `live_enrich(name, anchor_year=)` makes an explicit year outrank any name ratio and fetches 5 candidates per API so the right-year entry is found even when search ranks a namesake first. `create_game_dataset.py` still has the old rule — use `add_games.py` for targeted fixes rather than re-running the batch.
- **`audit_catalog.py`** — offline, read-only: classifies the ~250-probe `famous_games.py` list against the catalog (VISIBLE/HIDDEN/SPARSE/SUSPECT/MISSING), writes `data/catalog_audit.json` + **`data/preload_pins.json`** (matched catalog names that `app._preload_names` unions into the dropdown, so famous games can't be hidden by a weak blended score). Human-review the report before repairing.
- **`add_games.py`** — library + CLI. The CLI replays the 2026-07 manual triage; the library (`Catalog`, `fetch_game`, `refresh_row`, `append_games`, `save_catalog`, `reimpute_tail`) is what `update_games.py` builds on. Refreshes fill **only empty/zero** games.csv fields; appends add rows to `games.csv` + `E`/index/`Z` using **frozen transforms** (saved scalers from `embedding_scalers.pkl`, `action_pca.pkl` for Z), then `save_catalog` auto-runs `reimpute_tail` so new games' tags/collab blocks are kNN-imputed from their nearest text neighbors (global-mean imputation made every added game equidistant in 2 of 4 blocks → title-token junk in the neighborhoods). **This append is the sanctioned exception to "changing E invalidates `policy.pt`"**: rows below `N_PROTECTED=26120` stay byte-identical (asserted before saving), so no retrain. Same-name games get a ` (YYYY)` suffix (e.g. `God of War (2018)`). Idempotent; one-time `.bak` backups in `data/`.
- **`update_games.py`** — the recurring updater; safe to run any time. Per run: refreshes the sparsest rows (`--refresh`, default 50, from a ~16k backlog at coverage ≤ 5; 30-day per-game retry state in `data/update_state.json`), discovers popular recent releases via `RAWG.list_recent` (`--discover`, default 25, released in the last `--window-days` gated by `--min-added`), and takes manual `--add "Title:YYYY"` entries (fuzzy+year-deduped against the catalog). Newly appended games are pinned in `data/preload_pins_updater.json` (unioned by `app._preload_names` alongside the audit's `preload_pins.json`). `--deploy` uploads the changed artifacts to the Space; otherwise re-upload manually (games.csv, matrix, index, Z, updater pins).

### Policy behavior — the actor collapses to the embedding centroid (known limitation)
The trained IQL policy's predicted *action vector* is roughly state-independent — it converges toward the mean E-vector (advantage-weighted regression on a 1584-d continuous action with weakly-contrasting rewards). Consequence: **recommendation quality comes from the candidate generator's profile rerank, not the policy.** `recommend(...)` defaults to `profile_prefilter=True` for exactly this reason. **Do not remove the profile-rerank default or lower its priority assuming the policy ranks well on its own** — verify against `source/scripts/test_policy.ipynb` (cold-start vs per-profile divergence) first. Mitigations tried (more steps; per-user z-scored rewards) and remaining levers (lower-dim action target; discrete action-ID ranking) are tracked in `PLAN.md` → "Known limitations & follow-up strategies".

### `source/snippets/` — not used at runtime
`mdp_generation.py` and `train_step.py` are the original CQL sketches, now **superseded** by the built `scripts/build_mdp_dataset.py` (Stage 2) and `scripts/train_iql.py` (Stage 3) — kept only for history (they still reference CQL and toy data). `collaborative_embedding.py` is an older draft superseded by `scripts/generate_collaborative_embeding.py`. Don't import from `snippets/`.

### `source/create_game_table.ipynb` — legacy
First attempt that joined static Kaggle CSVs (`multi_decade_video_game_review_dataset.csv`, `hltb_dataset.csv`, `metacritics_games.csv`, `rawg_games_dataset.csv`). Superseded by the API-driven `scripts/create_game_dataset.py`. Kept for history; don't add to it.

## Data conventions

- All artifacts live under `data/`. `.csv` and `.pkl` are gitignored except `gamespot_reviews.csv` and `metacritic_reviews.csv` (intentional allow-list in `.gitignore`).
- `games.csv` columns mirror the `GameType` dataclass field names; `real_name` is the original query string and `name` is the canonical title chosen by the merge. List-typed columns are stored as `str(list)` and parsed back with `ast.literal_eval` (see `generate_game_tags_embedding.py:parse_tags`).
- `reviews.csv` schema after merge: `author, score, game_name, date` (+ `type, platform` only for Metacritic rows).
