"""
Web search tool for looking up best practices, documentation, and examples.
"""

try:
    from duckduckgo_search import DDGS
except ImportError:
    try:
        from ddgs import DDGS
    except ImportError:
        DDGS = None


def web_search(query: str, num_results: int = 5) -> list[dict]:
    """
    Search the web for relevant information.
    Returns list of {title, url, snippet}.
    """
    if DDGS is None:
        return [{"error": "duckduckgo-search not installed. pip install duckduckgo-search"}]
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
        return [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in results
        ]
    except ImportError:
        return [{"error": "duckduckgo-search not installed. pip install duckduckgo-search"}]
