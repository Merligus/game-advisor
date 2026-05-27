"""Stage 5: turn a user state + filters into ranked game recommendations.

`recommend(state, filters, played_games, top_n=5)` is the hot path the
HuggingFace app calls per request:

  1. candidate set  — `candidate_generator.candidates(filters, ...)`
  2. policy action  — run the TorchScript policy on `state` (dim 1584)
  3. rerank         — cosine of every candidate's embedding vs the action
  4. drop played    — remove games already in `played_games`
  5. enrich         — attach release_year / cover_url / description from games.csv
                      (optionally refreshed live from IGDB)

Candidate generation **profile-reranks by default** (`profile_prefilter=True`):
the candidate generator keeps the `candidate_k` (default 30) games closest to
the user's play history, and the policy reranks those. This anchors results to
the play history and masks the policy's current undertraining. With no play
history (cold start) it falls back to filter-only with no cap, so the policy
reranks the whole filtered set (no alphabetical-truncation artifact). Set
`profile_prefilter=False` to always let the policy rank the full filtered set —
the "trust the policy" mode, preferable once the policy is well trained.

(The profile rerank is why "Battlefield Hardline: Getaway" drops out of the RPG
demo: it passes the PC/2010+ filter but ranks 593rd in raw similarity to an RPG
history, outside the kept candidates — even though the policy's action points
near it. See PLAN.md Stage 5.)

`state` is supplied by the caller (Stage 6 `state_builder.cold_start_state`);
`played_games` must be canonical names (the resolved output of the state
builder) so the drop-played and optional profile rerank line up.
"""

import ast

import numpy as np

from app import artifacts
from app.candidate_generator import candidates


def _policy_action(state: np.ndarray) -> np.ndarray:
    import torch

    policy = artifacts.policy()
    t = torch.from_numpy(np.asarray(state, dtype=np.float32))
    if t.ndim == 1:
        t = t[None, :]
    with torch.no_grad():
        return policy(t).cpu().numpy().ravel()


def _first_cover(raw) -> str | None:
    """games.csv stores cover_url as a stringified list; return the first usable URL."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    try:
        urls = ast.literal_eval(raw) if isinstance(raw, str) else raw
    except (ValueError, SyntaxError):
        return None
    for u in urls or []:
        if u:
            return ("https:" + u) if str(u).startswith("//") else str(u)
    return None


def _enrich(name: str, score: float, igdb=None) -> dict:
    meta = artifacts.games_metadata()
    idx = artifacts.index_frame()
    row = idx.iloc[artifacts.name_to_row()[name]]
    rec = {
        "name": name,
        "score": score,
        "release_year": None if np.isnan(row["release_year"]) else int(row["release_year"]),
        "platforms": list(row["platforms"]),
        "cover_url": None,
        "description": None,
    }
    if name in meta.index:
        m = meta.loc[name]
        rec["cover_url"] = _first_cover(m.get("cover_url"))
        desc = m.get("description")
        rec["description"] = None if (desc is None or (isinstance(desc, float) and np.isnan(desc))) else str(desc)
    # Optional live cover-art refresh (Stage 6 may pass an IGDB client).
    if igdb is not None and not rec["cover_url"]:
        try:
            hits = igdb.search(name, max_n=1)
            if hits and hits[0].cover_url:
                rec["cover_url"] = _first_cover(hits[0].cover_url)
        except Exception:
            pass
    return rec


def recommend(
    state: np.ndarray,
    filters: dict,
    played_games=None,
    top_n: int = 5,
    profile_prefilter: bool = True,
    candidate_k: int = 30,
    igdb=None,
) -> list[dict]:
    """Return up to `top_n` recommendation dicts for the given state + filters."""
    played = list(played_games or [])

    # 1. Candidate set.
    if profile_prefilter and played:
        cand_idx = candidates(filters, played_games=played, k=candidate_k)
    else:
        cand_idx = candidates(filters, played_games=None, k=None)
    if len(cand_idx) == 0:
        return []

    # 2-3. Policy action + cosine rerank over the candidate set.
    action = _policy_action(state)
    a_norm = action / max(float(np.linalg.norm(action)), 1e-12)
    En = artifacts.embedding_matrix_normalized()
    sims = En[cand_idx] @ a_norm
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
        results.append(_enrich(name, float(sims[pos]), igdb=igdb))
        if len(results) == top_n:
            break
    return results
