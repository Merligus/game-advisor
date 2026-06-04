import html
import re
import sys
from urllib.parse import urlencode

from APIs.api_types import GamespotType

BASE = "https://www.gamespot.com"


class ChallengeError(RuntimeError):
    """Raised when Cloudflare returns a bot-challenge instead of JSON."""


def _looks_like_challenge(status, body):
    return status in (403, 503) or "Just a moment" in (body or "")[:1000]


class _CffiFetcher:
    def __init__(self):
        from curl_cffi import requests as creq
        self.session = creq.Session(impersonate="chrome")
        self.session.headers.update({"Accept": "application/json", "Referer": f"{BASE}/"})

    def get_json(self, url, **params):
        r = self.session.get(url, params=params, timeout=30)
        if _looks_like_challenge(r.status_code, r.text):
            raise ChallengeError(f"Cloudflare challenge (HTTP {r.status_code}) on {url}")
        r.raise_for_status()
        return r.json()

    def close(self):
        try:
            self.session.close()
        except Exception:
            pass


class _BrowserFetcher:
    def __init__(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(headless=True)
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
        import json
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
            raise ChallengeError(f"Cloudflare still blocking (HTTP {status}).")
        if status != 200:
            raise RuntimeError(f"HTTP {status} on {url}\n{body[:200]}")
        return json.loads(body)

    def close(self):
        try:
            self.browser.close()
            self._pw.stop()
        except Exception:
            pass


class Gamespot:
    def __init__(self):
        self._fetcher = self._make_fetcher()

    def _make_fetcher(self):
        try:
            return _CffiFetcher()
        except ImportError:
            pass
        try:
            return _BrowserFetcher()
        except ImportError:
            raise ImportError(
                "Neither curl_cffi nor Playwright is installed.\n"
                "  pip install curl_cffi\n"
                "  # or: pip install playwright && playwright install chromium"
            )

    def close(self):
        try:
            self._fetcher.close()
        except Exception:
            pass

    def _get_json(self, url, **params):
        try:
            return self._fetcher.get_json(url, **params)
        except ChallengeError:
            if isinstance(self._fetcher, _BrowserFetcher):
                raise
            print("curl_cffi got a Cloudflare challenge; falling back to Playwright.", file=sys.stderr)
            self._fetcher.close()
            try:
                self._fetcher = _BrowserFetcher()
            except ImportError:
                raise ImportError(
                    "Cloudflare blocked and Playwright isn't installed.\n"
                    "  pip install playwright && playwright install chromium"
                )
            return self._fetcher.get_json(url, **params)

    @staticmethod
    def _clean(html_text):
        return html.unescape(re.sub(r"<[^>]+>", "", html_text or "")).strip()

    def _find_games(self, name, max_n):
        # Ask for more results than needed so filtering to subtype='games' still yields max_n
        results = self._get_json(
            f"{BASE}/wp-json/wp/v2/search",
            search=name,
            per_page=max(10, max_n),
        )
        return [r for r in results if r.get("subtype") == "games"][:max_n]

    def _fetch_game(self, search_item):
        return self._get_json(search_item["_links"]["self"][0]["href"])

    def _resolve_terms(self, game):
        out = {}
        for link in game.get("_links", {}).get("wp:term", []):
            tax = link.get("taxonomy")
            if tax == "author":
                continue
            try:
                terms = self._get_json(link["href"], per_page=100)
                out[tax] = [t["name"] for t in terms if t.get("name")]
            except (ChallengeError, RuntimeError):
                names = []
                for tid in game.get(tax, []):
                    try:
                        t = self._get_json(f"{BASE}/wp-json/wp/v2/{tax}/{tid}")
                        if t.get("name"):
                            names.append(t["name"])
                    except RuntimeError:
                        pass
                out[tax] = names
        return out

    def _fetch_media(self, game_id):
        try:
            items = self._get_json(
                f"{BASE}/wp-json/wp/v2/media", parent=game_id, per_page=100
            )
            return [m["source_url"] for m in items if m.get("source_url")]
        except RuntimeError:
            return []

    def search(self, game_name: str, max_n: int = 1) -> list[GamespotType]:
        hits = self._find_games(game_name, max_n)
        results = []
        for hit in hits:
            try:
                game = self._fetch_game(hit)
                taxonomies = self._resolve_terms(game)
                media_urls = self._fetch_media(game["id"])

                acf = game.get("acf") or {}

                # Fall back to featured_images thumbnail when media endpoint returns nothing
                if not media_urls:
                    fi = game.get("featured_images") or {}
                    if fi.get("2x"):
                        media_urls = [fi["2x"]]

                # Franchise names + ESRB content descriptors as keywords
                keywords = (
                    taxonomies.get("game_franchise", [])
                    + taxonomies.get("esrb_descriptors", [])
                )

                results.append(GamespotType(
                    name=self._clean(game.get("title", {}).get("rendered", "")),
                    release=acf.get("gamepost_release_date", ""),
                    description=self._clean(game.get("content", {}).get("rendered", "")),
                    genres=taxonomies.get("genre", []),
                    platforms=taxonomies.get("platform", []),
                    themes=taxonomies.get("theme", []),
                    cover_url=media_urls,
                    keywords=keywords,
                ))
            except Exception as e:
                print(f"Gamespot: skipping '{hit.get('title')}': {e}", file=sys.stderr)
        return results
