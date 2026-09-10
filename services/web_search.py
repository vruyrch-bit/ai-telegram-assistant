"""Key-free public web search with bounded results and a short memory cache."""
import asyncio
from collections import OrderedDict
from time import monotonic
from urllib.parse import urlsplit
from ddgs import DDGS

_cache = OrderedDict()


def _search(query):
    with DDGS(timeout=8) as search:
        return search.text(query, max_results=5, backend='duckduckgo,bing', safesearch='moderate')


async def search_web(query):
    query = query.strip()
    if not query:
        raise ValueError('Search query cannot be blank.')
    cached = _cache.get(query)
    if cached and monotonic() - cached[0] < 120:
        return cached[1]
    try:
        raw = await asyncio.wait_for(asyncio.to_thread(_search, query), timeout=20)
    except Exception:
        raise ValueError('Free web search is temporarily unavailable. Try again later; no paid fallback is used.') from None
    results = []
    seen = set()
    for item in raw:
        url = item.get('href', '')
        parsed = urlsplit(url)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname or url in seen:
            continue
        seen.add(url)
        results.append({'title': item.get('title', '')[:300], 'url': url,
                        'snippet': item.get('body', '')[:1500]})
        if len(results) == 5:
            break
    _cache[query] = (monotonic(), results)
    _cache.move_to_end(query)
    if len(_cache) > 100:
        _cache.popitem(last=False)
    return results
