"""Web search and page fetching.

Search has pluggable backends. DuckDuckGo's "lite" endpoint needs no API key
and is the default so the harness works out of the box; the keyed backends are
better and switch on automatically when their key is in the environment.
"""

import html
import json
import os
import re
import urllib.parse
from html.parser import HTMLParser

from . import ToolError, tool
from ..cache import FetchError, fetch, fetch_json
from ..config import active

DDG_LITE = "https://lite.duckduckgo.com/lite/"
RESULT_LINK_RE = re.compile(
    r"<a[^>]+href=\"(https?://[^\"]+)\"[^>]*class='result-link'[^>]*>(.*?)</a>",
    re.DOTALL)
SNIPPET_RE = re.compile(r"<td class='result-snippet'>(.*?)</td>", re.DOTALL)


def _clean(fragment):
    """Strip tags and entities out of a snippet of markup."""
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


class _TextExtractor(HTMLParser):
    """Reduce a page to readable text, dropping scripts, styles and nav chrome."""

    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer"}
    BREAK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self._skip_depth = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag in self.BREAK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth and data.strip():
            self.parts.append(data.strip() + " ")

    def text(self):
        joined = "".join(self.parts)
        joined = re.sub(r"[ \t]+", " ", joined)
        return re.sub(r"\n\s*\n+", "\n\n", joined).strip()


@tool
def search_web(query: str, max_results: int = 6) -> str:
    """Search the web and return ranked titles, URLs and snippets.

    Use this for context a data API cannot give you: news of an early or severe
    season, a dominant subtype, school closures, a CDC FluView commentary.
    Follow up with fetch_url to read anything that looks worth reading.

    Args:
        query: The search query.
        max_results: How many results to return (1-15).
    """
    max_results = max(1, min(int(max_results), 15))
    backend = active().search_backend
    if backend == "duckduckgo" and os.environ.get("TAVILY_API_KEY"):
        backend = "tavily"          # prefer a real API when a key is present

    try:
        if backend == "tavily":
            results = _tavily(query, max_results)
        elif backend == "brave":
            results = _brave(query, max_results)
        elif backend == "serper":
            results = _serper(query, max_results)
        else:
            results = _duckduckgo(query, max_results)
    except FetchError as e:
        raise ToolError(f"search failed: {e}")

    if not results:
        return f"No results for {query!r}. Try different or broader terms."

    lines = [f"Search results for {query!r} (via {backend}):", ""]
    for i, (title, url, snippet) in enumerate(results, 1):
        lines.append(f"[{i}] {title}\n    {url}")
        if snippet:
            lines.append(f"    {snippet}")
    return "\n".join(lines)


def _duckduckgo(query, max_results):
    body = fetch(DDG_LITE, data=urllib.parse.urlencode({"q": query}).encode(),
                 timeout=20, ttl=active().cache_ttl,
                 cache_dir=active().cache_dir)
    links = RESULT_LINK_RE.findall(body)
    snippets = SNIPPET_RE.findall(body)
    results = []
    for i, (url, title) in enumerate(links[:max_results]):
        snippet = _clean(snippets[i]) if i < len(snippets) else ""
        results.append((_clean(title), url, snippet))
    return results


def _tavily(query, max_results):
    key = os.environ["TAVILY_API_KEY"]
    data = fetch_json(
        "https://api.tavily.com/search", timeout=30,
        headers={"Content-Type": "application/json"},
        data=json.dumps({"api_key": key, "query": query,
                         "max_results": max_results}).encode())
    return [(r.get("title", ""), r.get("url", ""), r.get("content", "")[:300])
            for r in data.get("results", [])]


def _brave(query, max_results):
    key = os.environ["BRAVE_API_KEY"]
    url = ("https://api.search.brave.com/res/v1/web/search?"
           + urllib.parse.urlencode({"q": query, "count": max_results}))
    data = fetch_json(url, timeout=30,
                      headers={"X-Subscription-Token": key,
                               "Accept": "application/json"})
    return [(r.get("title", ""), r.get("url", ""), _clean(r.get("description", "")))
            for r in data.get("web", {}).get("results", [])]


def _serper(query, max_results):
    key = os.environ["SERPER_API_KEY"]
    data = fetch_json(
        "https://google.serper.dev/search", timeout=30,
        headers={"X-API-KEY": key, "Content-Type": "application/json"},
        data=json.dumps({"q": query, "num": max_results}).encode())
    return [(r.get("title", ""), r.get("link", ""), r.get("snippet", ""))
            for r in data.get("organic", [])]


@tool
def fetch_url(url: str, max_chars: int = 4000) -> str:
    """Fetch a web page or file and return its text.

    HTML is reduced to readable text. JSON and CSV come back as-is, so this
    doubles as a way to pull a data file the built-in tools do not cover.

    Args:
        url: Absolute http(s) URL.
        max_chars: Truncate the extracted text to this many characters.
    """
    if not url.lower().startswith(("http://", "https://")):
        raise ToolError("url must start with http:// or https://")
    try:
        body = fetch(url, timeout=30, ttl=active().cache_ttl,
                     cache_dir=active().cache_dir)
    except FetchError as e:
        raise ToolError(str(e))

    stripped = body.lstrip()
    if stripped.startswith(("{", "[")) or "<" not in stripped[:200]:
        text = body                      # JSON, CSV or plain text: leave alone
    else:
        parser = _TextExtractor()
        parser.feed(body)
        text = parser.text()

    max_chars = max(200, min(int(max_chars), 20000))
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated; page has {len(text)} chars]"
    return f"# {url}\n\n{text}"
