"""Recurring catalog updater — run it any time to keep the games DB current.

Every run, within per-run budgets:

  1. REFRESH (--refresh, default 50): picks the sparsest catalog rows
     (coverage ascending, popularity descending), re-fetches them through the
     year-anchored live merge, and fills ONLY empty/zero games.csv fields.
  2. DISCOVER (--discover, default 25): asks RAWG for games *released* in the
     last --window-days (default 365) ordered by popularity, gated by
     --min-added (default 50) to skip shovelware; whatever isn't in the
     catalog yet is fetched, validated and appended to games.csv + E/index/Z
     with the frozen-transform machinery (policy stays valid, no retrain).
  3. MANUAL ADDS (--add "Title:YYYY", repeatable): processed like discoveries.

Newly appended games get kNN tags/collab imputation automatically and are
pinned into the dropdown via data/preload_pins_updater.json (unioned by
app._preload_names alongside the audit's pins) so fresh games are searchable
despite thin early metadata.

A state file (data/update_state.json) records per-game attempt dates; a game
isn't re-attempted for RETRY_DAYS, so repeated runs walk through the sparse
backlog instead of hammering the same rows. Idempotent and rate-limit polite
(1s per live fetch).

After a run that changed anything, re-upload the artifacts to the Space
(--deploy does it for you: games.csv, matrix, index, Z, updater pins).

Run from project root:
  python source/scripts/update_games.py --refresh 50 --discover 25
  python source/scripts/update_games.py --add "Hades II:2025" --refresh 0 --discover 0
"""

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from thefuzz import fuzz

from add_games import (Catalog, append_games, fetch_game, refresh_row,
                       save_catalog)
from recommender.enrichment import _normalize, _numbers, _year, live_enrich

DATA = ROOT / "data"
STATE_FILE = DATA / "update_state.json"
UPDATER_PINS = DATA / "preload_pins_updater.json"
RETRY_DAYS = 30
RATIO_MIN = 0.9   # keep in sync with audit_catalog.py
YEAR_TOL = 1
SPARSE_MAX_COVERAGE = 5  # rows at or below this are refresh candidates


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=1, sort_keys=True))


def recently_attempted(state: dict, name: str) -> bool:
    ts = state.get(name)
    if not ts:
        return False
    try:
        return (datetime.now() - datetime.fromisoformat(ts)).days < RETRY_DAYS
    except ValueError:
        return False


def coverage_series(games: pd.DataFrame, names: set) -> pd.Series:
    """Source-coverage 0-8 per catalog game — same signals as the audit/preload."""
    df = games.drop_duplicates(subset=["name"]).set_index("name")
    df = df[df.index.isin(names)]
    cov = (
        (df["cover_url"].fillna("[]") != "[]").astype(int)
        + df["description"].notna().astype(int)
    )
    for col in ("metacritic_rating", "user_rating", "rawg_rating",
                "igdb_rating", "hltb_rating", "main_story"):
        cov = cov + (pd.to_numeric(df[col], errors="coerce").fillna(0) > 0).astype(int)
    return cov


def refresh_pass(cat: Catalog, state: dict, budget: int) -> int:
    if budget <= 0:
        return 0
    names = cat.existing
    cov = coverage_series(cat.games, names)
    meta = cat.games.drop_duplicates(subset=["name"]).set_index("name")
    year_of = dict(zip(cat.idx["name"], cat.idx["release_year"]))

    candidates = cov[cov <= SPARSE_MAX_COVERAGE].index.tolist()
    ur = pd.to_numeric(meta["user_rating"], errors="coerce").fillna(0)
    candidates.sort(key=lambda n: (cov[n], -float(ur.get(n, 0.0))))

    done = 0
    print(f"\n=== REFRESH pass (budget {budget}; backlog {len(candidates)} rows at coverage<={SPARSE_MAX_COVERAGE}) ===")
    for name in candidates:
        if done >= budget:
            break
        if recently_attempted(state, name):
            continue
        y = year_of.get(name)
        anchor = int(y) if (y is not None and not (isinstance(y, float) and np.isnan(y))) else None
        d = live_enrich(name, anchor_year=anchor)
        time.sleep(1)
        state[name] = datetime.now().isoformat(timespec="seconds")
        rows = cat.games.index[cat.games["name"] == name]
        if len(rows) == 0:
            continue
        filled = refresh_row(cat.games, rows[0], d)
        done += 1
        print(f"  [{done}/{budget}] {name!r} (cov {int(cov[name])}): "
              f"filled {filled if filled else 'nothing'}")
    return done


def _catalog_matcher(cat: Catalog):
    """Fuzzy matcher against the catalog, reusing the hardened normalize /
    sequel-number logic and the audit's thresholds. Returns the matched catalog
    name (truthy) or None — callers that only need existence use truthiness;
    the --add path uses the name to refresh + pin the hidden row."""
    names = cat.idx["name"].tolist()
    years = cat.idx["release_year"].tolist()
    norm = [_normalize(n) for n in names]
    nums = [_numbers(n) for n in names]

    def known(title: str, year: int | None) -> str | None:
        # Prefer a year-CONFIRMED match over a year-unknown one: a franchise can
        # have both a dated row ('Fable (2004)') and an undated stub ('Fable'),
        # and the dated one is the intended target when the query carries a year.
        tn, tnum = _normalize(title), _numbers(title)
        best_dated, best_undated = None, None  # ((norm_ratio, raw_ratio), name)
        for i, n in enumerate(norm):
            if nums[i] != tnum:
                continue
            r = fuzz.ratio(tn, n) / 100.0
            if r < RATIO_MIN:
                continue
            # Raw-string ratio breaks normalized ties: edition-stripping makes
            # 'Fable Anniversary' and 'Fable' both normalize to "fable", and
            # the query "Fable Anniversary" must match the former.
            key = (r, fuzz.ratio(title.lower(), names[i].lower()) / 100.0)
            cy = years[i]
            if cy is None or (isinstance(cy, float) and np.isnan(cy)):
                if best_undated is None or key > best_undated[0]:
                    best_undated = (key, names[i])
            elif year is None or abs(int(cy) - year) <= YEAR_TOL:
                if best_dated is None or key > best_dated[0]:
                    best_dated = (key, names[i])
        if best_dated:
            return best_dated[1]
        if best_undated:
            return best_undated[1]
        return None

    return known


def discovery_pass(cat: Catalog, state: dict, budget: int, window_days: int, min_added: int) -> list[dict]:
    if budget <= 0:
        return []
    from APIs.rawg_api import RAWG

    end = date.today()
    start = end - timedelta(days=window_days)
    print(f"\n=== DISCOVERY pass (budget {budget}; released {start}..{end}, min_added {min_added}) ===")
    recent = RAWG().list_recent(str(start), str(end), max_results=budget * 6, min_added=min_added)
    print(f"  RAWG returned {len(recent)} popular recent releases")

    known = _catalog_matcher(cat)
    specs = []
    for g in recent:
        if len(specs) >= budget:
            break
        title, released = g.get("name"), g.get("released")
        year = _year(released)
        if not title or year is None:
            continue
        if recently_attempted(state, title):
            continue
        if known(title, year):
            continue
        state[title] = datetime.now().isoformat(timespec="seconds")
        specs.append({"query": title, "year": year})
        print(f"  new: {title!r} ({released}, added={g.get('added')})")
    if not specs:
        print("  nothing new — catalog already has the popular recent releases")
    return specs


def update_updater_pins(new_names: list[str]):
    try:
        pins = set(json.loads(UPDATER_PINS.read_text()))
    except (FileNotFoundError, ValueError):
        pins = set()
    pins.update(new_names)
    UPDATER_PINS.write_text(json.dumps(sorted(pins), indent=2))
    print(f"updater pins: {len(pins)} names ({UPDATER_PINS.name})")


def deploy():
    from huggingface_hub import HfApi
    api = HfApi()
    files = ["data/games.csv", "data/game_embeddings_matrix.npy",
             "data/game_embeddings_index.pkl", "data/game_actions_reduced.npy",
             "data/preload_pins_updater.json"]
    for f in files:
        if not (ROOT / f).exists():
            continue
        print(f"  upload {f}", flush=True)
        api.upload_file(path_or_fileobj=str(ROOT / f), path_in_repo=f,
                        repo_id="merligus/game-advisor", repo_type="space")
    print("Space rebuilding")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", type=int, default=50, help="sparse rows to refresh this run")
    ap.add_argument("--discover", type=int, default=25, help="new popular releases to add this run")
    ap.add_argument("--add", action="append", default=[], metavar="TITLE:YYYY",
                    help="manually add a game (repeatable)")
    ap.add_argument("--window-days", type=int, default=365, help="discovery release window")
    ap.add_argument("--min-added", type=int, default=50, help="RAWG popularity gate")
    ap.add_argument("--deploy", action="store_true", help="upload changed artifacts to the Space")
    args = ap.parse_args()

    cat = Catalog()
    state = load_state()
    print(f"catalog: {cat.N_start} games | state: {len(state)} attempts recorded")

    n_refreshed = refresh_pass(cat, state, args.refresh)

    specs = discovery_pass(cat, state, args.discover, args.window_days, args.min_added)
    known = _catalog_matcher(cat) if args.add else None
    pinned_existing = []
    for entry in args.add:
        title, _, year = entry.rpartition(":")
        if not title or not year.strip().isdigit():
            print(f"  !! --add {entry!r}: expected 'Title:YYYY', skipped")
            continue
        title, y = title.strip(), int(year)
        match = known(title, y)
        if match:
            # The user explicitly asked for this game: it exists but may be
            # hidden (sparse metadata below the preload cutoff). Refresh its
            # row — allowing a substantially longer description to replace a
            # stub — and pin it so it becomes searchable in the dropdown.
            print(f"  == --add {title!r} ({y}): already in catalog as {match!r} — refreshing + pinning")
            d = live_enrich(match, anchor_year=y)
            time.sleep(1)
            rows = cat.games.index[cat.games["name"] == match]
            if len(rows):
                filled = refresh_row(cat.games, rows[0], d, improve_description=True)
                print(f"     filled {filled if filled else 'nothing'}")
            pinned_existing.append(match)
            continue
        specs.append({"query": title, "year": y})

    added = []
    if specs:
        print(f"\n=== ADD pass ({len(specs)} candidates) ===")
        added = append_games(cat, specs)

    changed = bool(n_refreshed or added or pinned_existing)
    if changed:
        save_catalog(cat)  # asserts protected prefix; auto kNN-reimputes adds
        if added or pinned_existing:
            update_updater_pins(added + pinned_existing)
    save_state(state)

    print(f"\n=== summary: {n_refreshed} refreshed, {len(added)} added, "
          f"{len(pinned_existing)} existing pinned ===")
    if changed:
        if args.deploy:
            print("deploying to the Space...")
            deploy()
        else:
            print("artifacts changed — re-upload to the Space (or rerun with --deploy)")


if __name__ == "__main__":
    main()
