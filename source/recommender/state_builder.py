"""Stage 6 helper: build the user's cold-start state vector from played games.

`cold_start_state(played_games)` resolves each (possibly user-typed) title to a
canonical catalog `name` via fuzzy matching, then returns the mean of those
games' E-vectors — the analog of the running-average state the MDP used — plus
the list of names it actually resolved. Empty / unrecognized input yields a zero
vector (true cold start). This is the state the policy consumes at inference.
"""

import numpy as np
from thefuzz import fuzz

from recommender import artifacts


def resolve_name(query: str, min_ratio: int = 80) -> str | None:
    """Map a user-typed title to the closest canonical catalog name, or None."""
    n2r = artifacts.name_to_row()
    if query in n2r:
        return query
    q = query.lower()
    best, best_score = None, -1
    for name in n2r:
        s = fuzz.ratio(q, name.lower())
        if s > best_score:
            best, best_score = name, s
    return best if best_score >= min_ratio else None


def cold_start_state(played_games) -> tuple[np.ndarray, list[str]]:
    """Return (state_vector, resolved_names).

    state = mean of the E-vectors of the resolved games (zero vector if none).
    """
    E = artifacts.embedding_matrix()
    n2r = artifacts.name_to_row()
    rows, resolved = [], []
    for g in played_games or []:
        canon = g if g in n2r else resolve_name(g)
        if canon is not None and canon not in resolved:
            rows.append(n2r[canon])
            resolved.append(canon)
    if not rows:
        return np.zeros(E.shape[1], dtype=np.float32), resolved
    return E[rows].mean(axis=0).astype(np.float32), resolved
