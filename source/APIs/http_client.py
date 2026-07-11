"""Shared HTTP plumbing for the requests-based API wrappers (RAWG, IGDB,
Metacritic). Fixes the failure modes the original one-off wrappers shipped
with: no timeout on any call (infinite-hang risk), no retry on 429/5xx, and a
fresh TCP connection per request.

Gamespot has its own fetcher stack (curl_cffi / Playwright with Cloudflare
challenge handling) and HLTB goes through howlongtobeatpy — neither uses this.

Behavior: retries with exponential backoff on 429, 5xx and connection errors;
any other HTTP error (or exhausted retries) raises — callers (enrichment,
create_game_dataset) already catch per-source exceptions and degrade.
"""

import time

import requests

TIMEOUT = 20  # seconds, applied to every request
RETRIES = 3
BACKOFF = 2.0  # seconds; doubles per retry (2, 4, 8)
RETRY_STATUSES = {429, 500, 502, 503, 504}


class HttpClient:
    """A small requests.Session wrapper: timeout + retry/backoff + reuse."""

    def __init__(self, headers: dict | None = None):
        self.session = requests.Session()
        if headers:
            self.session.headers.update(headers)

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", TIMEOUT)
        last_exc = None
        for attempt in range(RETRIES):
            try:
                resp = self.session.request(method, url, **kwargs)
            except (requests.ConnectionError, requests.Timeout) as e:
                last_exc = e
                time.sleep(BACKOFF * (2 ** attempt))
                continue
            if resp.status_code in RETRY_STATUSES and attempt < RETRIES - 1:
                # Honor Retry-After when the API provides one (429s often do).
                retry_after = resp.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else BACKOFF * (2 ** attempt)
                except ValueError:
                    delay = BACKOFF * (2 ** attempt)
                time.sleep(min(delay, 60.0))
                continue
            resp.raise_for_status()
            return resp
        raise last_exc if last_exc else requests.ConnectionError(f"retries exhausted for {url}")

    def get_json(self, url: str, **kwargs) -> dict | list:
        return self.request("GET", url, **kwargs).json()

    def post(self, url: str, **kwargs) -> requests.Response:
        return self.request("POST", url, **kwargs)
