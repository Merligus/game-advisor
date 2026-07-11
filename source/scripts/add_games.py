"""Catalog repair library + CLI: add missing games / refresh sparse rows, then
append new games to the runtime artifacts without invalidating the trained
policy.

Library (imported by update_games.py):
  Catalog          in-memory handle on games.csv + E/index/Z + frozen transforms
  ensure_backups() one-time .bak copies in data/
  fetch_game()     live_enrich(query, anchor_year) + validation -> dict | None
  refresh_row()    fill ONLY empty/zero fields of an existing games.csv row
  append_games()   new games.csv rows + E/index/Z rows via FROZEN transforms
  save_catalog()   persist with byte-identity asserts on the protected prefix
  reimpute_tail()  upgrade appended rows' tags/collab blocks to kNN imputation

Frozen-transform invariant: rows 0..N_PROTECTED-1 are referenced by
mdp_dataset.npz / were present at training time; every save asserts they are
byte-identical, so policy.pt, the MDP and the PCA action space stay valid — no
retrain. New rows: title/description = live SBERT; tags/collab = kNN over text
neighbors (via reimpute_tail, run automatically after appends); scalars = saved
median/p99/min-max params.

CLI (unchanged behavior):
  python add_games.py                    # the 2026-07 manual triage repair
  python add_games.py --reimpute-tail K  # re-impute the last K appended rows

Run from project root.
"""

import json
import pickle
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))

import numpy as np
import pandas as pd

from recommender.enrichment import _year, live_enrich

DATA = ROOT / "data"
SCALAR_COLS = [
    "metacritic_rating", "igdb_rating", "rawg_rating", "hltb_rating",
    "user_rating", "main_story", "main_extra", "completionist",
]
TTB_COLS = {"main_story", "main_extra", "completionist"}
LIST_COLS = ["platforms", "cover_url", "developers", "publishers",
             "language_supports", "genres", "keywords"]
MAX_PLAUSIBLE_YEAR = 2035  # keep in sync with build_combined_embeddings.py

# Block layout of a combined-embedding row (build_combined_embeddings.py).
_TITLE = slice(0, 768)
_DESC = slice(768, 1536)
_TEXT = slice(0, 1536)   # title+desc (both L2'd) — the kNN similarity space
_TAGS = slice(1536, 1568)
_COLLAB = slice(1568, 1576)
# Rows below this index are referenced by mdp_dataset.npz / were present at
# training time — they must never be rewritten (policy validity).
N_PROTECTED = 26120


def _l2(v):
    return (v / max(float(np.linalg.norm(v)), 1e-12)).astype(np.float32)


def _empty(v) -> bool:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return True
    if isinstance(v, str):
        return v.strip() in ("", "[]", "nan")
    return False


class Catalog:
    """In-memory handle on every artifact the repair path touches."""

    def __init__(self):
        self.idx = pd.read_pickle(DATA / "game_embeddings_index.pkl")
        self.E = np.load(DATA / "game_embeddings_matrix.npy")
        self.Z = np.load(DATA / "game_actions_reduced.npy")
        with open(DATA / "embedding_scalers.pkl", "rb") as fh:
            self.scalers = pickle.load(fh)
        with open(DATA / "action_pca.pkl", "rb") as fh:
            self.pca = pickle.load(fh)
        self.games = pd.read_csv(DATA / "games.csv")
        self.N_start = len(self.E)  # size when loaded — the save-time assert prefix
        self.added: list[tuple[str, dict]] = []  # (final_name, merged_dict)

    @property
    def existing(self) -> set:
        return set(self.idx["name"])


def ensure_backups():
    for f in ("games.csv", "game_embeddings_matrix.npy",
              "game_embeddings_index.pkl", "game_actions_reduced.npy"):
        bak = DATA / (f + ".bak")
        if not bak.exists():
            shutil.copy2(DATA / f, bak)


def fetch_game(query: str, year: int, sleep: float = 1.0) -> dict | None:
    """Year-anchored live merge + validation. None when the fetch can't be
    trusted (no description, or the resolved release contradicts the year)."""
    d = live_enrich(query, anchor_year=year)
    if sleep:
        time.sleep(sleep)
    ry = _year(d["release"])
    if not d["description"] or ry is None or abs(ry - year) > 1:
        return None
    return d


def refresh_row(games: pd.DataFrame, i: int, d: dict) -> list[str]:
    """Fill only empty/zero fields of games.csv row i from the live merge d."""
    filled = []
    for col in SCALAR_COLS:
        cur = games.at[i, col]
        if (pd.isna(cur) or float(cur) <= 0) and d.get(col, 0) > 0:
            games.at[i, col] = float(d[col])
            filled.append(col)
    for col in LIST_COLS:
        if _empty(games.at[i, col]) and d.get(col):
            games.at[i, col] = str(d[col])
            filled.append(col)
    for col in ("release", "description"):
        if _empty(games.at[i, col]) and d.get(col):
            games.at[i, col] = str(d[col])
            filled.append(col)
    return filled


def append_games(cat: Catalog, specs: list[dict], verbose: bool = True) -> list[str]:
    """Append validated new games to games.csv + E/index/Z (in memory).

    Each spec: {"query": str, "year": int, "name": optional forced final name,
    "merged": optional pre-fetched live_enrich dict}. Name collisions get a
    " (YYYY)" suffix; unresolvable ones are skipped. Call save_catalog() after.
    """
    from sentence_transformers import SentenceTransformer
    import torch

    model = None  # lazy — only load SBERT if something validates
    existing = cat.existing
    added_names = []
    new_E, new_idx_rows = [], []
    N_base = len(cat.E)

    for spec in specs:
        final_hint = spec.get("name") or spec["query"]
        if final_hint in existing:
            if verbose:
                print(f"  == {final_hint!r} already in index, skipped")
            continue
        d = spec.get("merged") or fetch_game(spec["query"], spec["year"])
        if d is None:
            if verbose:
                print(f"  !! {spec['query']!r} ({spec['year']}): fetch failed validation, skipped")
            continue
        final = spec.get("name") or d["name"] or spec["query"]
        if final in existing:
            final = f"{final} ({spec['year']})"
            if final in existing:
                if verbose:
                    print(f"  !! {spec['query']!r}: name collision unresolvable, skipped")
                continue

        # games.csv row (exact column order; lists stored as str(list)).
        row = {c: "" for c in cat.games.columns}
        row.update({"real_name": spec["query"], "name": final,
                    "release": d["release"], "description": d["description"] or ""})
        for col in SCALAR_COLS:
            row[col] = float(d.get(col) or 0.0)
        for col in LIST_COLS:
            row[col] = str(d.get(col) or [])
        cat.games.loc[len(cat.games)] = row

        # Embedding row with frozen transforms (tags/collab start as the global
        # block means; reimpute_tail upgrades them to kNN right after saving).
        if model is None:
            model = SentenceTransformer("all-mpnet-base-v2",
                                        device="cuda" if torch.cuda.is_available() else "cpu")
        bm = cat.scalers["block_means"]
        sp = cat.scalers["scalars"]
        title_vec = _l2(model.encode(final))
        desc_vec = _l2(model.encode(d["description"]))
        scal = np.zeros(len(SCALAR_COLS), dtype=np.float32)
        for j, col in enumerate(SCALAR_COLS):
            p = sp[col]
            v = float(d.get(col) or 0.0)
            if v <= 0:
                v = p["median"]
            if col in TTB_COLS and p.get("p99"):
                v = min(v, p["p99"])
            x = (v - p["min"]) / (p["max"] - p["min"]) if p["max"] > p["min"] else 0.0
            scal[j] = min(max(x, 0.0), 1.0)
        scal /= np.sqrt(len(SCALAR_COLS))
        new_E.append(np.concatenate([title_vec, desc_vec, _l2(bm["tags"]), _l2(bm["collab"]), scal]))

        ry = _year(d["release"])
        new_idx_rows.append({
            "name": final, "row_idx": N_base + len(added_names),
            "release_year": float(ry) if (ry and ry <= MAX_PLAUSIBLE_YEAR) else np.nan,
            "platforms": list(d.get("platforms") or []),
            "language_supports": list(d.get("language_supports") or []),
            "main_story": float(d.get("main_story") or 0.0),
            "main_extra": float(d.get("main_extra") or 0.0),
            "completionist": float(d.get("completionist") or 0.0),
        })
        existing.add(final)
        added_names.append(final)
        cat.added.append((final, d))
        if verbose:
            print(f"  ++ {final!r} (release={d['release'][:10]}, "
                  f"user={d['user_rating']:.2f}, critic={d['metacritic_rating']:.2f})")

    if new_E:
        arr = np.asarray(new_E, dtype=np.float32)
        cat.E = np.vstack([cat.E, arr])
        cat.Z = np.vstack([cat.Z, cat.pca.transform(arr).astype(cat.Z.dtype)])
        cat.idx = pd.concat([cat.idx, pd.DataFrame(new_idx_rows)], ignore_index=True)
    return added_names


def save_catalog(cat: Catalog, reimpute: bool = True):
    """Persist all artifacts. The protected prefix (everything loaded at start,
    which always includes all training rows) must be byte-identical."""
    ensure_backups()
    E_old = np.load(DATA / "game_embeddings_matrix.npy", mmap_mode="r")
    assert np.array_equal(cat.E[:cat.N_start], E_old), "E prefix changed — aborting"
    assert cat.N_start >= N_PROTECTED, "loaded catalog smaller than the protected set"

    np.save(DATA / "game_embeddings_matrix.npy", cat.E)
    np.save(DATA / "game_actions_reduced.npy", cat.Z)
    cat.idx.to_pickle(DATA / "game_embeddings_index.pkl")
    cat.games.to_csv(DATA / "games.csv", index=False)
    n_added = len(cat.E) - cat.N_start
    print(f"saved: E {E_old.shape} -> {cat.E.shape}, games.csv {len(cat.games)} rows "
          f"(prefix byte-identical)")
    if reimpute and n_added > 0:
        print(f"\n=== kNN re-imputation of the {n_added} appended rows ===")
        reimpute_tail(n_added)


def reimpute_tail(k_tail: int, knn: int = 10):
    """Upgrade the tags/collab blocks of the last `k_tail` appended rows from
    the global-mean imputation to a kNN imputation: the mean of those blocks
    over the game's `knn` nearest *text* neighbors (title+desc cosine),
    skipping neighbors that are themselves mean-imputed. Global-mean blocks
    make every added game equidistant in 2 of 4 similarity blocks, which lets
    title-token junk ("War of the Worlds" for "God of War (2018)") crowd the
    neighborhoods; kNN blocks restore genre signal. Z rows are recomputed via
    the frozen PCA. Protected (training-time) rows are never touched.
    """
    E = np.load(DATA / "game_embeddings_matrix.npy")
    Z = np.load(DATA / "game_actions_reduced.npy")
    idx = pd.read_pickle(DATA / "game_embeddings_index.pkl")
    with open(DATA / "embedding_scalers.pkl", "rb") as fh:
        scalers = pickle.load(fh)
    with open(DATA / "action_pca.pkl", "rb") as fh:
        pca = pickle.load(fh)

    start = len(E) - k_tail
    assert start >= N_PROTECTED, f"refusing: tail reaches into protected rows (<{N_PROTECTED})"

    imput_tags = _l2(scalers["block_means"]["tags"])
    imput_collab = _l2(scalers["block_means"]["collab"])
    ref = E[:start]
    ref_text = ref[:, _TEXT]
    real_tags = ~np.all(np.isclose(ref[:, _TAGS], imput_tags, atol=1e-6), axis=1)
    real_collab = ~np.all(np.isclose(ref[:, _COLLAB], imput_collab, atol=1e-6), axis=1)
    names = idx["name"].tolist()

    E_new = E.copy()
    for r in range(start, len(E)):
        sims = ref_text @ E[r, _TEXT]
        order = np.argsort(-sims)
        nb_tags = [i for i in order if real_tags[i]][:knn]
        nb_collab = [i for i in order if real_collab[i]][:knn]
        if nb_tags:
            E_new[r, _TAGS] = _l2(ref[nb_tags, _TAGS].mean(axis=0))
        if nb_collab:
            E_new[r, _COLLAB] = _l2(ref[nb_collab, _COLLAB].mean(axis=0))
        print(f"  {names[r]!r}: tags<-{[names[i] for i in nb_tags[:4]]}")

    assert np.array_equal(E_new[:start], E[:start]), "protected rows changed — aborting"
    Z_new = Z.copy()
    Z_new[start:] = pca.transform(E_new[start:]).astype(Z.dtype)
    np.save(DATA / "game_embeddings_matrix.npy", E_new)
    np.save(DATA / "game_actions_reduced.npy", Z_new)
    print(f"reimputed rows {start}..{len(E)-1}; protected prefix byte-identical")

    En = E_new / np.maximum(np.linalg.norm(E_new, axis=1, keepdims=True), 1e-12)
    print("\n=== sanity: cosine top-5 per reimputed game ===")
    for r in range(start, len(E)):
        sims = En @ En[r]
        top = [names[i] for i in np.argsort(-sims) if i != r][:5]
        print(f"  {names[r]!r}: {top}")


# --- CLI: the 2026-07 manual triage repair (kept for the record / reruns) ----
ADD = [
    {"query": "Super Mario Galaxy", "year": 2007},
    {"query": "God of War II", "year": 2007},
    {"query": "Halo 3", "year": 2007},
    {"query": "Gears of War", "year": 2006},
    {"query": "Portal", "year": 2007},
    {"query": "Crysis", "year": 2007},
    {"query": "Star Wars: Knights of the Old Republic", "year": 2003},
    {"query": "Transistor", "year": 2014},
    {"query": "The Walking Dead", "year": 2012},
    {"query": "God of War", "year": 2018, "name": "God of War (2018)"},
    {"query": "Mass Effect", "year": 2007},
    {"query": "Metal Gear Solid", "year": 1998},
    {"query": "Resident Evil 2", "year": 2019, "name": "Resident Evil 2 (2019)"},
    {"query": "Bastion", "year": 2011},
    {"query": "Call of Duty: Modern Warfare 2", "year": 2009},
    {"query": "It Takes Two", "year": 2021, "name": "It Takes Two (2021)"},
]
# SUSPECTs that are probe-year artifacts (early-access years etc.), not damage.
SKIP_SUSPECTS = {"Mario Kart 8 Deluxe", "RimWorld", "Minecraft", "Stardew Valley",
                 "Slay the Spire", "Baldur's Gate 3", "Life is Strange"}
# Rows too ambiguous to refresh safely: the single 'DOOM' row is claimed by both
# the 1993 and 2016 probes and carries no year — leave it untouched.
SKIP_REFRESH_NAMES = {"DOOM"}


def main():
    audit = json.loads((DATA / "catalog_audit.json").read_text())
    cat = Catalog()
    ensure_backups()
    print(f"catalog N = {cat.N_start}")

    refreshes, seen = [], set()
    for e in audit:
        if e["class"] in ("SPARSE", "HIDDEN") and e["matched_name"] not in seen \
                and e["matched_name"] not in SKIP_REFRESH_NAMES:
            seen.add(e["matched_name"])
            refreshes.append((e["matched_name"], e["year"]))

    print(f"\n=== PHASE 1: refreshing {len(refreshes)} rows ===")
    for name, year in refreshes:
        d = live_enrich(name, anchor_year=year)
        time.sleep(1)
        rows = cat.games.index[cat.games["name"] == name]
        if len(rows) == 0:
            print(f"  !! {name!r}: not in games.csv, skipped")
            continue
        filled = refresh_row(cat.games, rows[0], d)
        print(f"  {name!r}: filled {filled if filled else 'nothing'}")

    print(f"\n=== PHASE 2+3: adding up to {len(ADD)} games ===")
    append_games(cat, ADD)
    save_catalog(cat)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--reimpute-tail":
        reimpute_tail(int(sys.argv[2]))
    else:
        main()
