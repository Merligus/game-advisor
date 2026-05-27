"""Stage 4: narrow the catalog to a shortlist of candidate games.

`candidates(filters, played_games=None, k=500)` returns row indices into the
embedding matrix E, filtered by the user's constraints and (optionally)
reranked by cosine similarity to the games they've already played. Stage 5
then runs the policy over this shortlist.

Filter semantics — **lenient on missing data**: a game is excluded only when
its metadata is present AND contradicts the filter. This matters because
language data covers only ~6% of the catalog and release year ~83%, so strict
matching on missing fields would empty the result set.

  year_min / year_max : release_year within [min, max]; unknown year passes.
  platforms           : list of platform names; a game passes if any of its
                        platforms case-insensitively matches any requested one
                        (substring either direction, so "PC" matches
                        "PC (Microsoft Windows)"). No platform data -> passes.
  language            : a single language substring; a game passes if any of
                        its languages contains it. No language data -> passes.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app import artifacts


def _platform_ok(game_platforms, requested_lower) -> bool:
    if not game_platforms:
        return True  # lenient on missing
    gp = [str(p).lower() for p in game_platforms]
    # Match if the requested label is a substring of a game platform label.
    # One-directional only: "pc" matches "pc (microsoft windows)" and "mac"
    # matches "macos", but a game labeled just "playstation" (PS1) must NOT
    # match a "playstation 5" request — the reverse direction caused that.
    return any(req in g for req in requested_lower for g in gp)


def _language_ok(game_langs, requested_lower) -> bool:
    if not game_langs:
        return True  # lenient on missing
    return any(requested_lower in str(lang).lower() for lang in game_langs)


def candidates(filters: dict, played_games=None, k: int | None = 500) -> np.ndarray:
    """Return ≤k row indices into E matching `filters`.

    If `played_games` (canonical names) is given, the matches are reranked by
    cosine to the mean embedding of those games before truncation; otherwise
    they're returned in catalog (row) order for the policy to rerank in Stage 5.
    """
    idx = artifacts.index_frame()
    mask = np.ones(len(idx), dtype=bool)

    # Year range — strict on present years, lenient on NaN.
    year = idx["release_year"].to_numpy(dtype=float)
    if filters.get("year_min") is not None:
        mask &= np.isnan(year) | (year >= filters["year_min"])
    if filters.get("year_max") is not None:
        mask &= np.isnan(year) | (year <= filters["year_max"])

    # Platforms.
    req_platforms = filters.get("platforms") or []
    if req_platforms:
        req_lower = [str(p).lower() for p in req_platforms]
        mask &= idx["platforms"].apply(lambda gp: _platform_ok(gp, req_lower)).to_numpy()

    # Language.
    lang = filters.get("language")
    if lang:
        lang_lower = str(lang).lower()
        mask &= idx["language_supports"].apply(lambda gl: _language_ok(gl, lang_lower)).to_numpy()

    cand_idx = np.where(mask)[0]

    # Optional cosine rerank against the played-games mean profile.
    if played_games:
        n2r = artifacts.name_to_row()
        rows = [n2r[g] for g in played_games if g in n2r]
        if rows and len(cand_idx) > 0:
            E = artifacts.embedding_matrix()
            profile = E[rows].mean(axis=0, keepdims=True)
            sims = cosine_similarity(profile, E[cand_idx])[0]
            cand_idx = cand_idx[np.argsort(-sims)]

    return cand_idx if k is None else cand_idx[:k]
