"""Live multi-API enrichment for recommended games — full GameType metadata.

`games.csv` (built by `scripts/create_game_dataset.py`) already merges every
GameType field, but it's sparse for the edition-suffixed / niche titles the
policy tends to surface. `live_enrich(name)` reconstructs the full record on
demand using the **same merge strategy as create_game_dataset.py**: query the
APIs, accept each result only if its name (and, when available, release year)
matches, then field-merge with per-field source priorities.

Source reality (measured): RAWG is the reliable workhorse (exact matches, rich
descriptions, covers via `background_image`); IGDB adds portrait box-art covers,
summaries, themes/modes/perspectives/languages — but its search is loose, so its
hits must pass the name guard; HLTB adds time-to-beat; Metacritic adds user score
+ devs/pubs; Gamespot is currently bot-walled (returns HTML, yields nothing).

Returns a plain dict with every GameType field. `lru_cache`d (games recur).
"""

import re
from functools import lru_cache

from thefuzz import fuzz

from APIs.api_types import (
    GamespotType,
    GameType,
    HLTBType,
    IGDBType,
    MetacriticType,
    RAWGType,
)

_clients = {}

GOOD_RATIO = 0.9
MIN_RATIO = 0.65
MIN_RATIO_NO_ANCHOR = 0.8  # stricter when we can't cross-check the release year


def _get_clients() -> dict:
    """Lazily build the API clients once (None if a client fails to construct)."""
    if not _clients:
        from APIs.hltb_api import HLTB
        from APIs.igdb_api import IGDB
        from APIs.metacritic_api import Metacritic
        from APIs.rawg_api import RAWG

        from APIs.gamespot_api import Gamespot

        for key, cls in (("rawg", RAWG), ("igdb", IGDB), ("hltb", HLTB), ("metacritic", Metacritic), ("gamespot", Gamespot)):
            try:
                _clients[key] = cls()
            except Exception:
                _clients[key] = None
    return _clients


# Edition/version qualifiers that inflate name similarity between different games
# ("CastleStorm - Definitive Edition" vs "Grim Dawn Definitive Edition"). Strip
# them before matching so the comparison is on the base title.
_EDITION_RE = re.compile(
    r"\b(definitive|complete|enhanced|ultimate|deluxe|platinum|special|anniversary|" r"legendary|gold|game of the year|goty|remastered|remaster|redux|" r"director'?s cut|bundle|collection)\b(\s+edition)?",
    re.IGNORECASE,
)


# Parenthetical years are disambiguation metadata, not part of the title:
# "God of War (2018)" must compare equal to "God of War" on the name axis
# (the year itself is checked separately via the release anchor).
_PAREN_YEAR_RE = re.compile(r"\(\s*(19|20)\d{2}\s*\)")

# Standalone roman-numeral tokens -> arabic, so "God of War II" vs
# "God of War III" (or "GTA V" vs "GTA 5") compare on the same number axis.
# i/v/x are included when they are whole tokens; the rare true-letter title
# ("Mega Man X") pays a small ambiguity cost for systematic franchise safety.
_ROMAN = {
    "i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7",
    "viii": "8", "ix": "9", "x": "10", "xi": "11", "xii": "12", "xiii": "13",
    "xiv": "14", "xv": "15", "xvi": "16", "xvii": "17", "xviii": "18",
    "xix": "19", "xx": "20",
}
_ROMAN_RE = re.compile(r"\b(" + "|".join(sorted(_ROMAN, key=len, reverse=True)) + r")\b")


def _normalize(s: str) -> str:
    s = _PAREN_YEAR_RE.sub(" ", (s or "").lower())
    s = _EDITION_RE.sub(" ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = _ROMAN_RE.sub(lambda m: _ROMAN[m.group(1)], s)
    return re.sub(r"\s+", " ", s).strip()


def _numbers(s: str) -> frozenset:
    """Numeric tokens of the normalized title — the sequel/edition number axis.

    Titles whose number sets differ are different games no matter how similar
    the words are: "god of war 2" != "god of war 3", "god of war" != "god of
    war 1", "final fantasy 10" != "final fantasy 10 2". High fuzzy ratios on
    numbered franchises are exactly how the original batch merge lost God of
    War II into the God of War III row.
    """
    return frozenset(re.findall(r"\d+", _normalize(s)))


def _name_ratio(a: str, b: str) -> float:
    """Fuzzy similarity on normalized base titles (edition suffixes stripped)."""
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return fuzz.ratio((a or "").lower(), (b or "").lower()) / 100.0
    return fuzz.ratio(na, nb) / 100.0


def _year(date_s) -> int | None:
    if not date_s or not isinstance(date_s, str) or len(date_s) < 4:
        return None
    try:
        return int(date_s[:4])
    except ValueError:
        return None


def _release_close(a, b) -> bool:
    ya, yb = _year(a), _year(b)
    return ya is not None and yb is not None and abs(ya - yb) <= 1


def _get_first_string(values):
    for v in values:
        if v is not None and len(str(v)) > 0:
            return str(v)
    return ""


def _get_first_float(values):
    for v in values:
        if v is not None and v > 0.0:
            return float(v)
    return 0.0


def _get_first_list(lists):
    for l in lists:
        if l:
            return [x for x in l if x]
    return []


def _get_union(lists):
    out = []
    for l in lists:
        for x in l or []:
            if x and x not in out:
                out.append(x)
    return out


@lru_cache(maxsize=4096)
def live_enrich(name: str, anchor_year: int | None = None) -> dict:
    """Full merged GameType metadata for `name` across all APIs (cached).

    `anchor_year` (repair/disambiguation mode): when given, it *is* the release
    anchor and a dated candidate must be within ±1 year of it — even a perfect
    name match can't override ("God of War" 2005 vs 2018 are name-identical).
    """
    clients = _get_clients()
    # Repair mode (anchor_year given) fetches a few candidates per API and keeps
    # the first that passes the guard: for name-identical pairs the top search
    # hit is often the *wrong* year (rankings favor the newest/most popular
    # entry), and top-1-only would silently drop the whole source. Runtime
    # enrichment keeps top-1 — cheap, and there's no year to arbitrate with.
    K = 5 if anchor_year is not None else 1

    def fetch(key, max_n):
        client = clients.get(key)
        if client is None:
            return []
        try:
            return client.search(name, max_n=max_n) or []
        except Exception:
            return []

    rawg_cands = fetch("rawg", K)
    igdb_cands = fetch("igdb", K)
    hltb_cands = fetch("hltb", K)
    meta_cands = fetch("metacritic", K)
    # Gamespot: always max_n=1 — each hit requires several follow-up requests
    # (fetch_game + resolve_terms + fetch_media), so max_n=5 would be ~40 calls.
    gs_cands = fetch("gamespot", 1)

    # Anchor the release year: an explicit anchor_year (repair mode) wins;
    # otherwise use the first trustworthy hit (RAWG first). The sequel-number
    # veto applies to anchor selection too — never anchor on a wrong sequel.
    if anchor_year is not None:
        anchor = f"{anchor_year}-01-01"
    else:
        anchor = ""
        firsts = [c[0] for c in (rawg_cands, igdb_cands, meta_cands, hltb_cands) if c]
        for cand in firsts:
            if _numbers(cand.name) != _numbers(name):
                continue
            if _name_ratio(cand.name, name) > 0.7 and cand.release:
                anchor = cand.release
                break

    def accept(obj, empty):
        if obj is None or not obj.name:
            return empty
        # Different sequel/edition numbers = different games, at any ratio.
        if _numbers(obj.name) != _numbers(name):
            return empty
        r = _name_ratio(obj.name, name)
        if anchor_year is not None:
            # Explicit year: dated candidates must match it; undated ones need
            # a stricter name match (we can't disprove them by year).
            if _year(obj.release) is None:
                return obj if r > MIN_RATIO_NO_ANCHOR else empty
            return obj if (r > MIN_RATIO and _release_close(obj.release, anchor)) else empty
        if r > GOOD_RATIO:
            return obj
        if anchor:
            if r > MIN_RATIO and _release_close(obj.release, anchor):
                return obj
        elif r > MIN_RATIO_NO_ANCHOR:
            return obj
        return empty

    def pick(cands, empty):
        for c in cands:
            got = accept(c, None)
            if got is not None:
                return got
        return empty

    rawg = pick(rawg_cands, RAWGType())
    igdb = pick(igdb_cands, IGDBType())
    hltb = pick(hltb_cands, HLTBType())
    meta = pick(meta_cands, MetacriticType())
    gs = pick(gs_cands, GamespotType())

    # Release: with an explicit anchor_year the anchor is synthetic (Jan 1), so
    # real accepted release dates win and the anchor is only a fallback;
    # otherwise the anchor IS a real date from the anchor-pick loop and leads.
    if anchor_year is not None:
        release = _get_first_string([g.release for g in (rawg, igdb, meta, hltb)] + [anchor])
    else:
        release = _get_first_string([anchor] + [g.release for g in (rawg, igdb, meta, hltb)])

    # Field-merge with the same source priorities as create_game_dataset.py.
    merged = GameType(
        real_name=name,
        name=_get_first_string([g.name for g in (gs, rawg, igdb, hltb, meta)]) or name,
        release=release,
        rawg_rating=rawg.rawg_rating,
        igdb_rating=igdb.igdb_rating,
        hltb_rating=hltb.hltb_rating,
        metacritic_rating=_get_first_float([meta.metacritic_rating, rawg.metacritic_rating]),
        user_rating=meta.user_rating,
        platforms=_get_first_list([igdb.platforms, rawg.platforms, hltb.platforms, meta.platforms]),
        main_story=_get_first_float([hltb.main_story, rawg.main_story]),
        main_extra=hltb.main_extra,
        completionist=hltb.completionist,
        cover_url=_get_union([igdb.cover_url, rawg.cover_url, gs.cover_url]),
        developers=_get_first_list([meta.developers, rawg.developers]),
        publishers=_get_first_list([meta.publishers, rawg.publishers]),
        description=_get_first_string([rawg.description, igdb.description, gs.description]),
        language_supports=igdb.language_supports,
        genres=_get_union([rawg.genres, igdb.genres, gs.genres, meta.genres]),
        keywords=_get_union(
            [
                rawg.keywords,
                igdb.keywords,
                igdb.themes,
                igdb.game_modes,
                igdb.player_perspectives,
                [rawg.esrb_rating],
                gs.themes,
                gs.keywords,
            ]
        ),
    )
    return merged.__dict__
