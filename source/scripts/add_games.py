"""Catalog repair: add missing famous games + refresh sparse rows, then append
the new games to the runtime artifacts without invalidating the trained policy.

Consumes data/catalog_audit.json (audit_catalog.py) plus the manual ADD triage
below. Two operations:

  REFRESH (SPARSE/HIDDEN rows): live_enrich(matched_name, anchor_year) and fill
  ONLY empty/zero fields of the existing games.csv row — never overwrites real
  data, never touches E/index (the row's embedding stays as built). Better
  preload score + richer cards, zero artifact churn.

  ADD (MISSING + corrupted-merge SUSPECTs): live_enrich(query, anchor_year) ->
  new games.csv row + new rows appended to the artifacts with FROZEN transforms:
    - title/description blocks: live SBERT (all-mpnet-base-v2), L2 per block;
    - tags/collab blocks: L2 of the saved per-block imputation means
      (embedding_scalers.pkl) — the SVD models/vocab were never persisted, and
      this matches the ~25% of existing rows that are mean-imputed;
    - scalars: saved median-impute -> p99 clip (ttb) -> min-max -> /sqrt(8).
  Existing E/Z rows stay byte-identical (asserted before saving), so policy.pt,
  mdp_dataset.npz and the PCA action space all remain valid — no retrain.

Run from project root. Idempotent: re-running skips names already in the index.
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

# --- Manual triage of catalog_audit.json (2026-07 session) -------------------
# MISSING probes plus the SUSPECTs that are real gaps (namesake or corrupted
# merge). "name" forces the stored catalog name where the merged name would
# collide with an existing different game.
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
# Their rows are correct; nothing to do. Kept here for the record.
SKIP_SUSPECTS = {"Mario Kart 8 Deluxe", "RimWorld", "Minecraft", "Stardew Valley",
                 "Slay the Spire", "Baldur's Gate 3", "Life is Strange"}
# Rows too ambiguous to refresh safely: the single 'DOOM' row is claimed by both
# the 1993 and 2016 probes and carries no year — leave it untouched.
SKIP_REFRESH_NAMES = {"DOOM"}


def _empty(v) -> bool:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return True
    if isinstance(v, str):
        return v.strip() in ("", "[]", "nan")
    return False


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


def main():
    audit = json.loads((DATA / "catalog_audit.json").read_text())

    idx = pd.read_pickle(DATA / "game_embeddings_index.pkl")
    E = np.load(DATA / "game_embeddings_matrix.npy")
    Z = np.load(DATA / "game_actions_reduced.npy")
    with open(DATA / "embedding_scalers.pkl", "rb") as fh:
        scalers = pickle.load(fh)
    with open(DATA / "action_pca.pkl", "rb") as fh:
        pca = pickle.load(fh)
    games = pd.read_csv(DATA / "games.csv")
    existing = set(idx["name"])
    N_old = len(E)

    # One-time backups.
    for f in ("games.csv", "game_embeddings_matrix.npy",
              "game_embeddings_index.pkl", "game_actions_reduced.npy"):
        bak = DATA / (f + ".bak")
        if not bak.exists():
            shutil.copy2(DATA / f, bak)
    print(f"backups in place; catalog N = {N_old}")

    # ---------------- PHASE 1: refresh sparse/hidden rows ----------------
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
        rows = games.index[games["name"] == name]
        if len(rows) == 0:
            print(f"  !! {name!r}: not in games.csv, skipped")
            continue
        filled = refresh_row(games, rows[0], d)
        print(f"  {name!r}: filled {filled if filled else 'nothing (row already full or fetch empty)'}")

    # ---------------- PHASE 2: add missing games ----------------
    print(f"\n=== PHASE 2: adding {len(ADD)} games ===")
    added = []  # (final_name, merged_dict)
    for spec in ADD:
        final_hint = spec.get("name") or spec["query"]
        if final_hint in existing:
            print(f"  == {final_hint!r} already in index, skipped (idempotent rerun)")
            continue
        d = live_enrich(spec["query"], anchor_year=spec["year"])
        time.sleep(1)
        ry = _year(d["release"])
        if not d["description"] or ry is None or abs(ry - spec["year"]) > 1:
            print(f"  !! {spec['query']!r} ({spec['year']}): fetch failed validation "
                  f"(desc={'Y' if d['description'] else 'n'}, release={d['release']!r}), skipped")
            continue
        final = spec.get("name") or d["name"] or spec["query"]
        if final in existing:
            final = f"{final} ({spec['year']})"
            if final in existing:
                print(f"  !! {spec['query']!r}: name collision unresolvable, skipped")
                continue
        row = {c: "" for c in games.columns}
        row.update({
            "real_name": spec["query"], "name": final, "release": d["release"],
            "description": d["description"] or "",
        })
        for col in SCALAR_COLS:
            row[col] = float(d.get(col) or 0.0)
        for col in LIST_COLS:
            row[col] = str(d.get(col) or [])
        games.loc[len(games)] = row
        existing.add(final)
        added.append((final, d))
        print(f"  ++ {final!r} (release={d['release'][:10]}, "
              f"user={d['user_rating']:.2f}, critic={d['metacritic_rating']:.2f})")

    # ---------------- PHASE 3: append artifacts for added games ----------------
    if added:
        print(f"\n=== PHASE 3: appending {len(added)} rows to E / index / Z ===")
        from sentence_transformers import SentenceTransformer
        import torch

        model = SentenceTransformer("all-mpnet-base-v2",
                                    device="cuda" if torch.cuda.is_available() else "cpu")

        def l2(v):
            return (v / max(float(np.linalg.norm(v)), 1e-12)).astype(np.float32)

        bm = scalers["block_means"]
        tags_vec = l2(bm["tags"])
        collab_vec = l2(bm["collab"])
        sp = scalers["scalars"]

        new_E, new_idx_rows = [], []
        for k, (final, d) in enumerate(added):
            title_vec = l2(model.encode(final))
            desc_vec = l2(model.encode(d["description"]))
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
            new_E.append(np.concatenate([title_vec, desc_vec, tags_vec, collab_vec, scal]))

            ry = _year(d["release"])
            new_idx_rows.append({
                "name": final, "row_idx": N_old + k,
                "release_year": float(ry) if (ry and ry <= MAX_PLAUSIBLE_YEAR) else np.nan,
                "platforms": list(d.get("platforms") or []),
                "language_supports": list(d.get("language_supports") or []),
                "main_story": float(d.get("main_story") or 0.0),
                "main_extra": float(d.get("main_extra") or 0.0),
                "completionist": float(d.get("completionist") or 0.0),
            })

        new_E = np.asarray(new_E, dtype=np.float32)
        E_out = np.vstack([E, new_E])
        Z_out = np.vstack([Z, pca.transform(new_E).astype(Z.dtype)])
        idx_out = pd.concat([idx, pd.DataFrame(new_idx_rows)], ignore_index=True)

        # Policy-validity proof: the pre-existing rows must be byte-identical.
        assert np.array_equal(E_out[:N_old], E), "E prefix changed — aborting"
        assert np.array_equal(Z_out[:N_old], Z), "Z prefix changed — aborting"
        assert list(idx_out["name"][:N_old]) == list(idx["name"]), "index prefix changed — aborting"

        np.save(DATA / "game_embeddings_matrix.npy", E_out)
        np.save(DATA / "game_actions_reduced.npy", Z_out)
        idx_out.to_pickle(DATA / "game_embeddings_index.pkl")
        print(f"  E: {E.shape} -> {E_out.shape} | Z: {Z.shape} -> {Z_out.shape} (prefixes byte-identical)")

        # Sanity: cosine top-5 neighbors of each added game (human eyeball gate).
        En = E_out / np.maximum(np.linalg.norm(E_out, axis=1, keepdims=True), 1e-12)
        names_all = idx_out["name"].tolist()
        print("\n=== sanity: cosine top-5 per added game ===")
        for k, (final, _) in enumerate(added):
            sims = En @ En[N_old + k]
            top = [names_all[i] for i in np.argsort(-sims)[1:6]]
            print(f"  {final!r}: {top}")

    games.to_csv(DATA / "games.csv", index=False)
    print(f"\ngames.csv written ({len(games)} rows). Done.")


if __name__ == "__main__":
    main()
