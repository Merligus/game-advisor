#!/usr/bin/env python3
"""
Query a game on GameSpot's WordPress REST API by name — Cloudflare-aware.

GameSpot sits behind Cloudflare bot protection. A plain `requests` call gets a
403 "Just a moment..." challenge page, because Cloudflare fingerprints the TLS
handshake (JA3), not just the User-Agent header. This script handles that with
two backends:

  1. curl_cffi  -> impersonates a real Chrome TLS fingerprint. Usually enough
                   for a content site's passive challenge.
                   (pip install curl_cffi)
  2. Playwright -> a real headless Chromium that actually solves the challenge,
                   then runs fetch() in-page for every call (reusing the
                   cf_clearance cookie). Used automatically if curl_cffi is
                   still challenged.
                   (pip install playwright && playwright install chromium)

Flow:
  1. /wp-json/wp/v2/search?search=<name>  -> first result with subtype 'games'
  2. follow _links.self[0].href           -> /wp-json/wp/v2/games/<id>
  3. resolve term IDs to names via _links["wp:term"] URLs (?post=<id>),
     falling back to single-term /wp-json/wp/v2/<tax>/<id>
  4. media via /wp-json/wp/v2/media?parent=<id>
  5. release date from acf.gamepost_release_date

Usage:
    pip install curl_cffi
    python gamespot_game.py The Legend of Zelda
    python gamespot_game.py "Elden Ring" --json
    python gamespot_game.py "Hades II" --browser --headful   # force real browser
"""

import argparse
import html
import json
import re
import sys
from urllib.parse import urlencode

BASE = "https://www.gamespot.com"


class ChallengeError(RuntimeError):
    """Raised when Cloudflare returns a bot-challenge instead of JSON."""


def _looks_like_challenge(status, body):
    return status in (403, 503) or "Just a moment" in (body or "")[:1000]


# --- backends --------------------------------------------------------------

class CffiFetcher:
    """Backend 1: curl_cffi impersonating Chrome's TLS fingerprint."""

    def __init__(self):
        from curl_cffi import requests as creq  # lazy import
        self.session = creq.Session(impersonate="chrome")
        self.session.headers.update({
            "Accept": "application/json",
            "Referer": f"{BASE}/",
        })

    def get_json(self, url, **params):
        r = self.session.get(url, params=params, timeout=30)
        if _looks_like_challenge(r.status_code, r.text):
            raise ChallengeError(f"Cloudflare challenge (HTTP {r.status_code}) on {url}")
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} on {url}\n{r.text[:200]}")
        return r.json()

    def close(self):
        try:
            self.session.close()
        except Exception:
            pass


class BrowserFetcher:
    """Backend 2: real Chromium (Playwright). Solves the challenge once, then
    runs fetch() inside the page for every JSON call."""

    def __init__(self, headless=True):
        from playwright.sync_api import sync_playwright  # lazy import
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(headless=headless)
        self.page = self.browser.new_page()
        self.page.goto(f"{BASE}/", wait_until="domcontentloaded", timeout=60000)
        self._wait_past_challenge()

    def _wait_past_challenge(self):
        for _ in range(30):
            try:
                if "Just a moment" not in self.page.title():
                    return
            except Exception:
                pass
            self.page.wait_for_timeout(1000)

    def get_json(self, url, **params):
        if params:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urlencode(params)}"
        result = self.page.evaluate(
            """async (u) => {
                const r = await fetch(u, {headers: {'Accept': 'application/json'}});
                return {status: r.status, body: await r.text()};
            }""",
            url,
        )
        status, body = result["status"], result["body"]
        if _looks_like_challenge(status, body):
            raise ChallengeError(
                f"Cloudflare still blocking (HTTP {status}). Try running with --headful.")
        if status != 200:
            raise RuntimeError(f"HTTP {status} on {url}\n{body[:200]}")
        return json.loads(body)

    def close(self):
        try:
            self.browser.close()
            self._pw.stop()
        except Exception:
            pass


# --- query logic (backend-agnostic) ----------------------------------------

_FETCHER = None


def get_json(url, **params):
    return _FETCHER.get_json(url, **params)


def clean(html_text):
    """Strip tags and unescape entities from WordPress 'rendered' fields."""
    return html.unescape(re.sub(r"<[^>]+>", "", html_text or "")).strip()


def find_game(name):
    results = get_json(f"{BASE}/wp-json/wp/v2/search", search=name, per_page=10)
    for item in results:
        if item.get("subtype") == "games":
            return item
    if results:
        seen = sorted({str(r.get("subtype")) for r in results})
        print(f"warning: no 'games' result; subtypes found: {', '.join(seen)}",
              file=sys.stderr)
    return None


def fetch_game(search_item):
    return get_json(search_item["_links"]["self"][0]["href"])


def resolve_term_by_id(taxonomy, term_id):
    """Single-term lookup, e.g. /wp-json/wp/v2/genre/588 -> 'Open-World'."""
    return get_json(f"{BASE}/wp-json/wp/v2/{taxonomy}/{term_id}").get("name")


def resolve_terms(game):
    """Resolve every taxonomy (genre, platform, theme, franchise, rating...)
    via the wp:term URLs the game JSON provides; fall back to per-ID lookups."""
    out = {}
    for link in game.get("_links", {}).get("wp:term", []):
        tax = link.get("taxonomy")
        if tax == "author":  # coauthors, not game info
            continue
        try:
            terms = get_json(link["href"], per_page=100)
            out[tax] = [t.get("name") for t in terms]
        except ChallengeError:
            raise
        except RuntimeError:
            names = []
            for tid in game.get(tax, []):
                try:
                    names.append(resolve_term_by_id(tax, tid))
                except RuntimeError:
                    names.append(f"#{tid}")
            out[tax] = names
    return out


def fetch_media(game_id):
    items = get_json(f"{BASE}/wp-json/wp/v2/media", parent=game_id, per_page=100)
    return [{"id": m.get("id"),
             "title": clean(m.get("title", {}).get("rendered", "")),
             "url": m.get("source_url")} for m in items]


def query_game(name):
    hit = find_game(name)
    if not hit:
        return None
    game = fetch_game(hit)
    return {
        "id": game["id"],
        "title": clean(game.get("title", {}).get("rendered", "")),
        "url": game.get("link"),
        "release_date": (game.get("acf") or {}).get("gamepost_release_date"),
        "description": clean(game.get("content", {}).get("rendered", "")),
        "taxonomies": resolve_terms(game),
        "media": fetch_media(game["id"]),
        "raw": game,
    }


# --- backend selection + CLI -----------------------------------------------

def make_primary(args):
    if args.browser:
        try:
            return BrowserFetcher(headless=not args.headful)
        except ImportError:
            sys.exit("Playwright isn't installed.\n"
                     "  pip install playwright\n  playwright install chromium")
    try:
        return CffiFetcher()
    except ImportError:
        print("curl_cffi not installed; using Playwright instead.", file=sys.stderr)
        try:
            return BrowserFetcher(headless=not args.headful)
        except ImportError:
            sys.exit("Neither curl_cffi nor Playwright is installed. Install one:\n"
                     "  pip install curl_cffi\n"
                     "  # or: pip install playwright && playwright install chromium")


def main():
    ap = argparse.ArgumentParser(description="Query a game on GameSpot by name")
    ap.add_argument("name", nargs="+", help="game name (quotes optional)")
    ap.add_argument("--json", action="store_true", help="dump everything as JSON")
    ap.add_argument("--browser", action="store_true",
                    help="skip curl_cffi and use a real browser (Playwright)")
    ap.add_argument("--headful", action="store_true",
                    help="show the browser window (more reliable vs Cloudflare)")
    args = ap.parse_args()
    name = " ".join(args.name)

    global _FETCHER
    info = None
    try:
        _FETCHER = make_primary(args)
        try:
            info = query_game(name)
        except ChallengeError as err:
            if isinstance(_FETCHER, BrowserFetcher):
                raise
            print(f"{err}\nFalling back to a real browser (Playwright)...",
                  file=sys.stderr)
            _FETCHER.close()
            try:
                _FETCHER = BrowserFetcher(headless=not args.headful)
            except ImportError:
                sys.exit("Cloudflare blocked the request and Playwright isn't "
                         "installed.\nInstall it:\n"
                         "  pip install playwright\n  playwright install chromium")
            info = query_game(name)
    except RuntimeError as err:
        sys.exit(f"Request failed:\n{err}")
    finally:
        if _FETCHER is not None:
            _FETCHER.close()

    if not info:
        sys.exit(f"No game found for '{name}'")

    if args.json:
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return

    tx = info["taxonomies"]

    def fmt(key):
        return ", ".join(tx.get(key, [])) or "-"

    print(f"\n{info['title']}  (id={info['id']})")
    print(f"URL:          {info['url']}")
    print(f"Release date: {info['release_date'] or '?'}")
    print(f"Genres:       {fmt('genre')}")
    print(f"Platforms:    {fmt('platform')}")
    print(f"Themes:       {fmt('theme')}")
    print(f"Franchise:    {fmt('game_franchise')}")
    print(f"Ratings:      {fmt('game_rating')}")
    print(f"\nDescription:\n{info['description'][:600]}")
    if info["media"]:
        print(f"\nMedia ({len(info['media'])}):")
        for m in info["media"][:5]:
            print(f"  - {m['title'] or m['id']}: {m['url']}")


if __name__ == "__main__":
    main()
