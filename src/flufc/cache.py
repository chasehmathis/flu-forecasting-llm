"""On-disk HTTP cache.

Surveillance APIs publish weekly, so refetching them on every agent step is
pure latency. Caching also means a run can be replayed later and see the same
data the model saw -- which matters when you are scoring forecasts after the
fact and want to know what was knowable at the time.
"""

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)


class FetchError(RuntimeError):
    """A request failed in a way the caller should report to the model."""


def _key(url, data):
    h = hashlib.sha256(url.encode())
    if data:
        h.update(b"\0")
        h.update(data)
    return h.hexdigest()[:32]


def fetch(url, *, data=None, headers=None, timeout=30, ttl=0, cache_dir=None,
          max_bytes=2_000_000):
    """GET (or POST, if `data`) a URL, returning text. Cached for `ttl` seconds.

    Raises FetchError on any transport or HTTP error, with a message that is
    safe to hand back to the model as a tool result.
    """
    path = None
    if ttl and cache_dir:
        path = Path(cache_dir) / (_key(url, data) + ".json")
        if path.exists():
            try:
                blob = json.loads(path.read_text())
                if time.time() - blob["fetched_at"] < ttl:
                    return blob["body"]
            except (json.JSONDecodeError, KeyError, OSError):
                pass  # corrupt entry: fall through and refetch

    request_headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
    request_headers.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=request_headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(max_bytes)
            charset = resp.headers.get_content_charset() or "utf-8"
    except urllib.error.HTTPError as e:
        raise FetchError(f"HTTP {e.code} from {url}") from e
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise FetchError(f"could not reach {url}: {e}") from e

    body = raw.decode(charset, "replace")

    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(
                {"url": url, "fetched_at": time.time(), "body": body}))
        except OSError:
            pass  # a cache we cannot write to is not a reason to fail the call

    return body


def fetch_json(url, **kwargs):
    """fetch() the URL and parse it as JSON."""
    body = fetch(url, **kwargs)
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise FetchError(f"{url} did not return JSON: {e}") from e
