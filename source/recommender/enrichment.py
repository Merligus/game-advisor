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


def _normalize(s: str) -> str:
    s = _EDITION_RE.sub(" ", (s or "").lower())
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


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


def _search(client, name, empty):
    if client is None:
        return empty
    try:
        res = client.search(name, max_n=1)
        return res[0] if res else empty
    except Exception:
        return empty


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
def live_enrich(name: str) -> dict:
    """Full merged GameType metadata for `name` across all APIs (cached)."""
    clients = _get_clients()
    rawg = _search(clients.get("rawg"), name, RAWGType())
    igdb = _search(clients.get("igdb"), name, IGDBType())
    hltb = _search(clients.get("hltb"), name, HLTBType())
    meta = _search(clients.get("metacritic"), name, MetacriticType())
    # Gamespot: call with max_n=1 — each hit requires several follow-up requests
    # (fetch_game + resolve_terms + fetch_media), so max_n=10 would be ~80 calls.
    gs_client = clients.get("gamespot")
    if gs_client is not None:
        try:
            gs_res = gs_client.search(name, max_n=1)
            gs = gs_res[0] if gs_res else GamespotType()
        except Exception:
            gs = GamespotType()
    else:
        gs = GamespotType()

    # Anchor the release year on the first trustworthy hit (RAWG is most reliable).
    anchor = ""
    for cand in (rawg, igdb, meta, hltb):
        if _name_ratio(cand.name, name) > 0.7 and cand.release:
            anchor = cand.release
            break

    def accept(obj, empty):
        if obj is None or not obj.name:
            return empty
        r = _name_ratio(obj.name, name)
        if r > GOOD_RATIO:
            return obj
        if anchor:
            if r > MIN_RATIO and _release_close(obj.release, anchor):
                return obj
        elif r > MIN_RATIO_NO_ANCHOR:
            return obj
        return empty

    rawg = accept(rawg, RAWGType())
    igdb = accept(igdb, IGDBType())
    hltb = accept(hltb, HLTBType())
    meta = accept(meta, MetacriticType())
    gs = accept(gs, GamespotType())

    # Field-merge with the same source priorities as create_game_dataset.py.
    merged = GameType(
        real_name=name,
        name=_get_first_string([g.name for g in (gs, rawg, igdb, hltb, meta)]) or name,
        release=_get_first_string([anchor] + [g.release for g in (rawg, igdb, meta, hltb)]),
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
