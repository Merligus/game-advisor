"""Audit the catalog against the famous-games probe list — offline, read-only.

For each (title, year) probe in famous_games.py, classify how the deployed app
would treat it, using only local data (no API calls):

  VISIBLE  in the catalog and in the dropdown's preload list
  HIDDEN   in the catalog, below the preload cutoff, decent coverage (>= 6)
  SPARSE   in the catalog, below the preload cutoff, sparse metadata (< 6)
           -> metadata-refresh candidate (refresh alone should lift it)
  SUSPECT  a same-name row exists but its year contradicts the probe (> 1 off)
           -> likely a corrupted merge (the God of War II -> III class) or a
              same-name namesake; repair adds the probe as a new, year-
              disambiguated entry
  MISSING  no catalog row matches -> repair adds it

Matching reuses the hardened guard from recommender.enrichment (_normalize /
_numbers): edition suffixes and parenthetical years stripped, roman numerals
mapped to arabic, and a sequel-number veto so "God of War" never matches
"God of War I". Fuzzy threshold 0.9 on normalized names; year tolerance ±1.

Output: console table + data/catalog_audit.json (consumed by add_games.py).
Run from project root. Human gate: review the report before repairing.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))            # for `import app` (preload list)
sys.path.insert(0, str(ROOT / "source"))  # for recommender.*
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for famous_games

import numpy as np
from thefuzz import fuzz

from famous_games import FAMOUS_GAMES
from recommender import artifacts
from recommender.enrichment import _normalize, _numbers

RATIO_MIN = 0.9
YEAR_TOL = 1
SPARSE_COVERAGE = 6


def _coverage_row(row) -> int:
    """Source-coverage score 0-8 — keep in sync with app._preload_names."""
    cov = 0
    cov += 1 if (isinstance(row.get("cover_url"), str) and row["cover_url"] != "[]") else 0
    cov += 1 if isinstance(row.get("description"), str) and row["description"] else 0
    for col in ("metacritic_rating", "user_rating", "rawg_rating", "igdb_rating", "hltb_rating", "main_story"):
        v = row.get(col)
        cov += 1 if (v is not None and not (isinstance(v, float) and np.isnan(v)) and float(v) > 0) else 0
    return cov


def main():
    print("Loading catalog + preload list (imports the app once)...")
    import app  # noqa: E402  — provides PRELOAD_NAMES with zero formula duplication

    idx = artifacts.index_frame()
    meta = artifacts.games_metadata()
    preload = set(app.PRELOAD_NAMES)

    names = idx["name"].tolist()
    years = idx["release_year"].tolist()
    norm = [_normalize(n) for n in names]
    nums = [_numbers(n) for n in names]

    results = []
    for title, year in FAMOUS_GAMES:
        tn, tnum = _normalize(title), _numbers(title)
        candidates = []  # (ratio, name, cat_year)
        for i, n in enumerate(names):
            if nums[i] != tnum:
                continue
            r = fuzz.ratio(tn, norm[i]) / 100.0
            if r >= RATIO_MIN:
                candidates.append((r, n, years[i]))

        entry = {"title": title, "year": year, "class": None, "matched_name": None,
                 "matched_year": None, "ratio": None, "coverage": None}
        if not candidates:
            entry["class"] = "MISSING"
        else:
            year_ok = [c for c in candidates
                       if (c[2] is None or (isinstance(c[2], float) and np.isnan(c[2]))
                           or abs(int(c[2]) - year) <= YEAR_TOL)]
            pool = year_ok if year_ok else candidates
            r, mname, myear = max(pool, key=lambda c: c[0])
            entry["matched_name"] = mname
            entry["matched_year"] = None if (myear is None or (isinstance(myear, float) and np.isnan(myear))) else int(myear)
            entry["ratio"] = round(r, 3)
            if not year_ok:
                entry["class"] = "SUSPECT"
            else:
                cov = _coverage_row(meta.loc[mname]) if mname in meta.index else 0
                entry["coverage"] = cov
                if mname in preload:
                    entry["class"] = "VISIBLE"
                elif cov >= SPARSE_COVERAGE:
                    entry["class"] = "HIDDEN"
                else:
                    entry["class"] = "SPARSE"
        results.append(entry)

    out = ROOT / "data" / "catalog_audit.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)

    # Preload pins: every famous probe that matched a catalog row, by catalog
    # name. app._preload_names unions these into the dropdown so a famous game
    # can never be hidden by a weak blended score (some — review-bombed or
    # metadata-poor rows — can't be lifted by any reasonable top-N cutoff).
    pins = sorted({e["matched_name"] for e in results if e["matched_name"]})
    pins_out = ROOT / "data" / "preload_pins.json"
    with open(pins_out, "w") as fh:
        json.dump(pins, fh, indent=2)
    print(f"preload pins ({len(pins)} names) written to {pins_out}")

    counts = {}
    for e in results:
        counts[e["class"]] = counts.get(e["class"], 0) + 1
    print(f"\n=== audit of {len(results)} probes ===")
    for cls in ("VISIBLE", "HIDDEN", "SPARSE", "SUSPECT", "MISSING"):
        print(f"  {cls:8s}: {counts.get(cls, 0)}")
    print(f"\nreport written to {out}\n")

    for cls in ("MISSING", "SUSPECT", "SPARSE", "HIDDEN"):
        rows = [e for e in results if e["class"] == cls]
        if not rows:
            continue
        print(f"--- {cls} ({len(rows)}) ---")
        for e in rows:
            extra = ""
            if e["matched_name"]:
                extra = f" -> matched {e['matched_name']!r} (year={e['matched_year']}, ratio={e['ratio']})"
            if e["coverage"] is not None:
                extra += f" cov={e['coverage']}"
            print(f"  {e['title']} ({e['year']}){extra}")
        print()


if __name__ == "__main__":
    main()
