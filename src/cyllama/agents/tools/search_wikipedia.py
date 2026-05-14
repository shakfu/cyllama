"""``search_wikipedia`` -- network IO with a bounded blast radius:
hardcoded endpoint, short timeout, capped response size, URL-encoded
query, no follow-up requests, no HTML parsing beyond stripping
``<span class="searchmatch">`` highlights.
"""

import json as _json
import re
import urllib.parse as _urlparse
import urllib.request as _urlrequest
from typing import Dict, List

from .core import tool


# Wikipedia search constants. Hardcoded so the tool can't be tricked into
# hitting an arbitrary host -- ``urllib.request.urlopen`` is happy to
# follow ``file://`` URLs if you let user input reach it.
_WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
_WIKI_TIMEOUT_SECONDS = 5.0
_WIKI_MAX_RESPONSE_BYTES = 64 * 1024  # 64 KiB is plenty for 3 snippets
_WIKI_USER_AGENT = "cyllama-demo-tools/1.0 (https://github.com/shakfu/cyllama)"
# Strip the highlight spans Wikipedia injects into ``snippet`` fields.
# We deliberately don't try to be a general HTML parser -- if the API
# starts returning richer markup we'd rather show it raw than silently
# drop information.
_WIKI_HIGHLIGHT_RE = re.compile(r"</?span[^>]*>")


@tool
def search_wikipedia(query: str, limit: int = 3) -> List[Dict[str, str]]:
    """Search English Wikipedia and return the top matching pages.

    Uses Wikipedia's official action API (``action=query&list=search``);
    no HTML scraping, no follow-up page fetches. The endpoint is
    hardcoded to ``en.wikipedia.org``; the request times out after 5
    seconds; the response is capped at 64 KiB.

    Args:
        query: Free-text search query. URL-encoded before transmission.
        limit: Number of results to return, in 1..10. Defaults to 3.

    Returns:
        A list of dicts, each with ``title``, ``snippet`` (HTML
        highlight spans stripped), and ``url`` (a stable wiki link).
        Empty list when the API returns no matches.
    """
    if not query.strip():
        raise ValueError("query must be non-empty")
    if not 1 <= limit <= 10:
        raise ValueError(f"limit must be in 1..10, got {limit}")

    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": str(limit),
        "srprop": "snippet",
    }
    url = f"{_WIKI_API_URL}?{_urlparse.urlencode(params)}"
    request = _urlrequest.Request(url, headers={"User-Agent": _WIKI_USER_AGENT})

    try:
        with _urlrequest.urlopen(request, timeout=_WIKI_TIMEOUT_SECONDS) as response:  # noqa: S310 - hardcoded https URL
            payload = response.read(_WIKI_MAX_RESPONSE_BYTES + 1)
    except Exception as e:  # noqa: BLE001 - surface any transport failure as a tool error
        raise RuntimeError(f"wikipedia request failed: {e}") from e

    if len(payload) > _WIKI_MAX_RESPONSE_BYTES:
        raise RuntimeError(f"wikipedia response exceeded {_WIKI_MAX_RESPONSE_BYTES} bytes; aborting")

    try:
        data = _json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, _json.JSONDecodeError) as e:
        raise RuntimeError(f"wikipedia returned non-JSON payload: {e}") from e

    results = data.get("query", {}).get("search", [])
    out: List[Dict[str, str]] = []
    for item in results:
        title = str(item.get("title", ""))
        snippet = _WIKI_HIGHLIGHT_RE.sub("", str(item.get("snippet", "")))
        url_title = _urlparse.quote(title.replace(" ", "_"))
        out.append(
            {
                "title": title,
                "snippet": snippet,
                "url": f"https://en.wikipedia.org/wiki/{url_title}",
            }
        )
    return out
