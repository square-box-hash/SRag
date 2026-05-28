# srag/search_providers.py
"""
Search provider abstraction for SRag.
Supports SearXNG (primary) and DDGS (fallback).
"""
from __future__ import annotations
import asyncio
import logging
import httpx
from typing import Optional

logger = logging.getLogger(__name__)

# ── Public SearXNG instances (fallback if no self-hosted) ─────────────────────
PUBLIC_SEARXNG_INSTANCES = [
    "https://searx.be",
    "https://searxng.world",
    "https://search.bus-hit.me",
    "https://searx.tiekoetter.com",
]


class SearchResult:
    """Unified search result across providers."""
    def __init__(self, url: str, title: str, snippet: str = "", engine: str = ""):
        self.url     = url
        self.title   = title
        self.snippet = snippet
        self.engine  = engine

    def to_dict(self) -> dict:
        return {
            "url":     self.url,
            "title":   self.title,
            "snippet": self.snippet,
            "engine":  self.engine,
        }


# ── SearXNG provider ──────────────────────────────────────────────────────────

class SearXNGProvider:
    """
    SearXNG search provider.
    Works with self-hosted or public instances.
    """

    def __init__(
        self,
        instance_url:  str            = "",
        timeout:       float          = 10.0,
        max_results:   int            = 10,
        engines:       list[str]      = None,
        language:      str            = "en",
    ):
        # Use provided instance, or try public ones
        self.instance_url = instance_url.rstrip("/") if instance_url else ""
        self.timeout      = timeout
        self.max_results  = max_results
        self.engines      = engines or ["google", "bing", "duckduckgo", "brave"]
        self.language     = language
        self._active_instance: Optional[str] = None

    async def _find_working_instance(self, client: httpx.AsyncClient) -> Optional[str]:
        """Try public instances and return first working one."""
        if self.instance_url:
            return self.instance_url

        for instance in PUBLIC_SEARXNG_INSTANCES:
            try:
                resp = await client.get(
                    f"{instance}/search",
                    params={"q": "test", "format": "json"},
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    logger.debug("SearXNG: using instance %s", instance)
                    return instance
            except Exception:
                continue

        return None

    async def search(self, query: str, max_results: int = None) -> list[SearchResult]:
        """Search via SearXNG JSON API."""
        k = max_results or self.max_results

        async with httpx.AsyncClient(
            timeout          = self.timeout,
            follow_redirects = True,
        ) as client:

            if not self._active_instance:
                self._active_instance = await self._find_working_instance(client)

            if not self._active_instance:
                logger.warning("SearXNG: no working instance found")
                return []

            try:
                resp = await client.get(
                    f"{self._active_instance}/search",
                    params={
                        "q":        query,
                        "format":   "json",
                        "engines":  ",".join(self.engines),
                        "language": self.language,
                        "pageno":   1,
                    },
                    headers={"Accept": "application/json"},
                )

                if resp.status_code != 200:
                    logger.warning(
                        "SearXNG: instance %s returned %d",
                        self._active_instance, resp.status_code
                    )
                    self._active_instance = None  # reset, try next time
                    return []

                data    = resp.json()
                results = data.get("results", [])

                return [
                    SearchResult(
                        url     = r.get("url", ""),
                        title   = r.get("title", ""),
                        snippet = r.get("content", ""),
                        engine  = r.get("engine", "searxng"),
                    )
                    for r in results[:k]
                    if r.get("url")
                ]

            except Exception as e:
                logger.warning("SearXNG search failed: %s", e)
                self._active_instance = None
                return []


# ── DDGS provider ─────────────────────────────────────────────────────────────

class DDGSProvider:
    """DuckDuckGo search provider via duckduckgo-search."""

    def __init__(self, max_results: int = 10, timeout: float = 10.0):
        self.max_results = max_results
        self.timeout     = timeout

    async def search(self, query: str, max_results: int = None) -> list[SearchResult]:
        """Search via DDGS."""
        k = max_results or self.max_results
        try:
            from duckduckgo_search import DDGS
            loop    = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: list(DDGS().text(query, max_results=k))
            )
            return [
                SearchResult(
                    url     = r.get("href", ""),
                    title   = r.get("title", ""),
                    snippet = r.get("body", ""),
                    engine  = "ddgs",
                )
                for r in results
                if r.get("href")
            ]
        except Exception as e:
            logger.warning("DDGS search failed: %s", e)
            return []


# ── Unified search layer ──────────────────────────────────────────────────────

class SearchLayer:
    """
    Priority-based search layer.
    SearXNG first, DDGS fallback.
    Deduplicates results across providers.
    """

    def __init__(
        self,
        searxng_instance: str       = "",
        searxng_engines:  list[str] = None,
        use_searxng:      bool      = True,
        use_ddgs:         bool      = True,
        max_results:      int       = 10,
        timeout:          float     = 10.0,
    ):
        self.use_searxng = use_searxng
        self.use_ddgs    = use_ddgs
        self.max_results = max_results

        self.searxng = SearXNGProvider(
            instance_url = searxng_instance,
            timeout      = timeout,
            max_results  = max_results,
            engines      = searxng_engines,
        ) if use_searxng else None

        self.ddgs = DDGSProvider(
            max_results = max_results,
            timeout     = timeout,
        ) if use_ddgs else None

    async def search(self, query: str, max_results: int = None) -> list[SearchResult]:
        """
        Search with priority: SearXNG → DDGS fallback.
        If SearXNG returns enough results, DDGS is skipped.
        If SearXNG fails or returns too few, DDGS fills the gap.
        """
        k       = max_results or self.max_results
        results = []
        seen    = set()

        def _dedup_add(new_results: list[SearchResult]) -> None:
            for r in new_results:
                if r.url and r.url not in seen:
                    seen.add(r.url)
                    results.append(r)

        # ── Priority 1: SearXNG ───────────────────────────────────────────
        if self.searxng:
            searxng_results = await self.searxng.search(query, max_results=k)
            _dedup_add(searxng_results)
            logger.debug(
                "SearXNG: %d results for '%s'",
                len(searxng_results), query[:40]
            )

        # ── Priority 2: DDGS (fallback or supplement) ─────────────────────
        if self.ddgs and len(results) < k:
            needed      = k - len(results)
            ddgs_results = await self.ddgs.search(query, max_results=needed + 3)
            _dedup_add(ddgs_results)
            logger.debug(
                "DDGS: %d results for '%s' (gap fill)",
                len(ddgs_results), query[:40]
            )

        logger.info(
            "SearchLayer: %d total results for '%s'",
            len(results), query[:40]
        )
        return results[:k]

    def status(self) -> dict:
        return {
            "searxng_enabled":  self.use_searxng,
            "ddgs_enabled":     self.use_ddgs,
            "active_instance":  getattr(
                self.searxng, "_active_instance", None
            ) if self.searxng else None,
        }