"""Stage 5: turn a user state + filters into ranked game recommendations.

`recommend(state, filters, played_games, top_n, mode)` is the hot path the
HuggingFace app calls per request. It runs three interchangeable ranking modes:

  - "profile_rl"   (default): candidate generator keeps the `candidate_k` games
                   closest to the play history, then the RL policy reranks those.
                   History-anchored; masks the policy's centroid collapse.
  - "rl_only":     policy ranks the entire filtered catalog ("trust the policy").
                   Exposes the policy's raw behavior (currently weak — see PLAN.md
                   "Known limitations").
  - "profile_only": rank purely by cosine to the play-history profile; the policy
                   is not used at all. A content-based baseline / ablation.

All modes share: filter → rank → drop played → enrich (year / cover / description
from games.csv, optionally refreshed live from IGDB).

`state` is supplied by the caller (Stage 6 `state_builder.cold_start_state`);
`played_games` must be canonical names (the resolved output of the state builder)
so the drop-played step and the profile ranking line up.
"""

import ast

import numpy as np

from recommender import artifacts
from recommender.candidate_generator import candidates

MODES = ("profile_rl", "rl_only", "profile_only")


def _policy_action(state: np.ndarray) -> np.ndarray:
    import torch

    policy = artifacts.policy()
    t = torch.from_numpy(np.asarray(state, dtype=np.float32))
    if t.ndim == 1:
        t = t[None, :]
    with torch.no_grad():
        return policy(t).cpu().numpy().ravel()


# All GameType fields, grouped by how they're parsed out of games.csv.
_STR_FIELDS = ("name", "release", "description")
_FLOAT_FIELDS = (
    "rawg_rating",
    "igdb_rating",
    "hltb_rating",
    "metacritic_rating",
    "user_rating",
    "main_story",
    "main_extra",
    "completionist",
)
_LIST_FIELDS = (
    "platforms",
    "cover_url",
    "developers",
    "publishers",
    "language_supports",
    "genres",
    "keywords",
)


def _parse_list(v) -> list:
    """games.csv stores list columns as str(list); parse back to a clean list."""
    if isinstance(v, (list, tuple)):
        return [x for x in v if x]
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    try:
        items = ast.literal_eval(v) if isinstance(v, str) else v
    except (ValueError, SyntaxError):
        return []
    return [x for x in items if x] if isinstance(items, (list, tuple)) else []


def _display_cover(cover_list) -> str | None:
    """First usable cover URL: fix protocol-relative URLs, upscale IGDB thumbnails."""
    for u in cover_list or []:
        if u:
            url = ("https:" + u) if str(u).startswith("//") else str(u)
            return url.replace("/t_thumb/", "/t_cover_big/")
    return None


def _enrich(name: str, score: float, enrich: bool = False) -> dict:
    """Build the full GameType record for `name` from games.csv, optionally
    filling empty fields from the live multi-API merge."""
    meta = artifacts.games_metadata()
    idx = artifacts.index_frame()
    row = idx.iloc[artifacts.name_to_row()[name]]

    rec: dict = {"score": score, "release_year": None if np.isnan(row["release_year"]) else int(row["release_year"])}
    for f in _STR_FIELDS:
        rec[f] = ""
    for f in _FLOAT_FIELDS:
        rec[f] = 0.0
    for f in _LIST_FIELDS:
        rec[f] = []
    rec["name"] = name

    if name in meta.index:
        m = meta.loc[name]
        for f in _STR_FIELDS:
            v = m.get(f)
            rec[f] = "" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
        rec["name"] = name  # keep the canonical index name
        for f in _FLOAT_FIELDS:
            v = m.get(f)
            try:
                rec[f] = float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else 0.0
            except (TypeError, ValueError):
                rec[f] = 0.0
        for f in _LIST_FIELDS:
            rec[f] = _parse_list(m.get(f))

    # Live multi-API fill for whatever games.csv is missing.
    needs = (not rec["description"]) or (not _display_cover(rec["cover_url"])) or (not rec["genres"])
    if enrich and needs:
        from recommender.enrichment import live_enrich

        extra = live_enrich(name)
        for f in _STR_FIELDS:
            if not rec[f]:
                rec[f] = extra.get(f) or ""
        for f in _FLOAT_FIELDS:
            if not rec[f]:
                rec[f] = extra.get(f) or 0.0
        for f in _LIST_FIELDS:
            if not rec[f]:
                rec[f] = extra.get(f) or []

    rec["cover_url_display"] = _display_cover(rec["cover_url"])
    return rec


def recommend(
    state: np.ndarray,
    filters: dict,
    played_games=None,
    top_n: int = 5,
    mode: str = "profile_rl",
    candidate_k: int = 30,
    enrich: bool = False,
    profile_prefilter: bool | None = None,
) -> list[dict]:
    """Return up to `top_n` recommendation dicts for the given state + filters.

    `mode` is one of MODES. `profile_prefilter` is a legacy alias kept for older
    callers (True -> "profile_rl", False -> "rl_only"); when given it overrides `mode`.
    """
    if profile_prefilter is not None:
        mode = "profile_rl" if profile_prefilter else "rl_only"
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    played = list(played_games or [])
    n2r = artifacts.name_to_row()
    played_rows = [n2r[g] for g in played if g in n2r]
    E = artifacts.embedding_matrix()
    En = artifacts.embedding_matrix_normalized()

    # 1. Candidate set.
    if mode == "profile_rl" and played_rows:
        cand_idx = candidates(filters, played_games=played, k=candidate_k)
    elif mode == "profile_only" and played_rows:
        cand_idx = candidates(filters, played_games=played, k=None)
    else:
        # rl_only, or a profile mode with no usable history -> whole filtered set.
        cand_idx = candidates(filters, played_games=None, k=None)
    if len(cand_idx) == 0:
        return []

    # 2-3. Score the candidates.
    if mode == "profile_only" and played_rows:
        # Pure content-based: cosine to the play-history mean profile, no policy.
        profile = E[played_rows].mean(axis=0)
        ref = profile / max(float(np.linalg.norm(profile)), 1e-12)
    else:
        # Policy modes (and profile_only with no history -> degrade to policy).
        action = _policy_action(state)
        ref = action / max(float(np.linalg.norm(action)), 1e-12)
    sims = En[cand_idx] @ ref
    order = np.argsort(-sims)

    # 4-5. Drop played, take top_n, enrich.
    names = artifacts.index_frame()["name"].values
    played_set = set(played)
    results = []
    for pos in order:
        row = int(cand_idx[pos])
        name = names[row]
        if name in played_set:
            continue
        results.append(_enrich(name, float(sims[pos]), enrich=enrich))
        if len(results) == top_n:
            break
    return results
