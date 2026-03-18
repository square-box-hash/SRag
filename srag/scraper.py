import asyncio
from datetime import datetime
import random

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
import trafilatura


REALISTIC_HEADERS = [
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    },
    {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) "
            "Version/17.0 Safari/605.1.15"
        ),
        "Accept-Language": "en-GB,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    },
]

# CHANGE 1: Expanded exception tuple — covers all common httpx failure modes
FETCH_ERRORS = (
    httpx.HTTPStatusError,
    httpx.RequestError,
    httpx.TimeoutException,
    httpx.ConnectError,
)

class AnuInfrastructureScraper:
    def __init__(
        self,
        max_results: int = 3,
        timeout: float = 5.0,
        max_chars: int = 2000,
        extract_mode: str = "basic",
    ):
        self.max_results = max_results
        self.timeout = timeout
        self.max_chars = max_chars
        self.extract_mode = extract_mode

        if extract_mode not in ["basic", "trafilatura"]:
            raise ValueError("extract_mode must be 'basic' or 'trafilatura'")

    async def get_facts(self, query: str):
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=self.max_results))

        urls = [(r["href"], r["title"]) for r in results]

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [
                self._fetch_and_clean(client, url, title)
                for url, title in urls
            ]
            docs = await asyncio.gather(*tasks, return_exceptions=True)

        # CHANGE 2: Log any unexpected exceptions that slipped through
        slate_entries = []
        for d in docs:
            if isinstance(d, dict):
                slate_entries.append(d)
            elif isinstance(d, Exception):
                print(f"⚠️  Unhandled exception during scrape: {type(d).__name__}: {d}")

        return slate_entries

    async def search(self, query: str) -> list[dict]:
        docs = await self.get_facts(query)
        results = []
        for i, d in enumerate(docs):
            results.append({
                "title": d["title"],
                "url": d["source"],
                "content": d["content"][: self.max_chars],
                "raw_content": d["content"],
                "score": 1.0 - i * 0.1,
                "published_date": d.get("timestamp"),
                "author": d.get("author"),
            })
        return results

    async def _fetch_and_clean(
        self,
        client: httpx.AsyncClient,
        url: str,
        title: str,
    ):
        # CHANGE 3: Wrap entire method in try/except — returns None on any failure
        # None is filtered out by the isinstance(d, dict) check in get_facts
        try:
            headers = random.choice(REALISTIC_HEADERS)
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            html = resp.text
        except FETCH_ERRORS as e:
            print(f"⚠️  Fetch failed [{url}]: {type(e).__name__}: {e}")
            return None

        # CHANGE 4: Trafilatura fallback — if it returns empty/short text,
        # fall back to BS4 instead of indexing a useless empty chunk
        try:
            if self.extract_mode == "trafilatura":
                soup = BeautifulSoup(html, "html.parser")
                clean_text = trafilatura.extract(html) or ""
                if len(clean_text.strip()) < 100:
                    # Trafilatura failed (JS-heavy, paywall, etc.) — use BS4
                    print(f"⚠️  Trafilatura got <100 chars for {url}, falling back to BS4")
                    for el in soup(["nav", "footer", "script", "style"]):
                        el.decompose()
                    clean_text = soup.get_text(separator=" ", strip=True)
            else:
                soup = BeautifulSoup(html, "html.parser")
                for element in soup(["nav", "footer", "script", "style"]):
                    element.decompose()
                clean_text = soup.get_text(separator=" ", strip=True)
        except Exception as e:
            print(f"⚠️  Parsing failed [{url}]: {e}")
            return None

        clean_text = clean_text[: self.max_chars]

        # CHANGE 5: Skip docs that are still empty after both extraction attempts
        if not clean_text.strip():
            print(f"⚠️  Empty content after extraction, skipping: {url}")
            return None

        page_title = self._extract_title(soup) or title
        page_date = self._extract_date(soup)
        page_author = self._extract_author(soup)
        page_image = self._extract_image(soup)

        return {
            "source": url,
            "title": page_title,
            "content": clean_text,
            "timestamp": page_date or datetime.utcnow().isoformat(),
            "author": page_author,
            "image": page_image,
        }

    def _extract_title(self, soup: BeautifulSoup) -> str | None:
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            return h1.get_text(strip=True)
        return None

    def _extract_date(self, soup: BeautifulSoup) -> str | None:
        meta = (
            soup.find("meta", {"property": "article:published_time"})
            or soup.find("meta", {"name": "date"})
        )
        if meta and meta.get("content"):
            return meta["content"]
        return None

    def _extract_author(self, soup: BeautifulSoup) -> str | None:
        meta = soup.find("meta", {"name": "author"})
        if meta and meta.get("content"):
            return meta["content"]
        return None

    def _extract_image(self, soup: BeautifulSoup) -> str | None:
        meta = soup.find("meta", {"property": "og:image"})
        if meta and meta.get("content"):
            return meta["content"]
        return None