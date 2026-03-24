import asyncio
from datetime import datetime
import random
import time

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
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    },
    {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) "
            "Version/17.0 Safari/605.1.15"
        ),
        "Accept-Language": "en-GB,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    },
    {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    },
]


BLOCKED_DOMAINS = {
    "indianexpress.com",
    "hindustantimes.com",
    "ndtv.com",
    "timesofindia.indiatimes.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "nytimes.com",
    "theatlantic.com",
    "wired.com",
    "businessinsider.com",
    "forbes.com",
    "udemy.com",
    "coursera.org",
    "medium.com",
    "levelup.gitconnected.com",
    "datacamp.com",
    "youtube.com",
    "youtu.be",
    "en.wikipedia.org",
    "wikipedia.org",
}


PRIORITY_DOMAINS = {
    "docs.python.org",
    "fastapi.tiangolo.com",
    "realpython.com",
    "geeksforgeeks.org",
    "stackoverflow.com",
    "github.com",
    "wikipedia.org",
    "docs.lancedb.com",
    "arxiv.org",
    "pypi.org",
}


FETCH_ERRORS = (
    httpx.HTTPStatusError,
    httpx.RequestError,
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
)


def _get_domain(url: str) -> str:
    try:
        from urllib.parse import urlparse

        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def _is_blocked(url: str) -> bool:
    domain = _get_domain(url)
    return any(blocked in domain for blocked in BLOCKED_DOMAINS)


def _sort_urls(urls: list[tuple]) -> list[tuple]:
    """Prioritize high quality domains, filter blocked ones."""
    filtered = [(url, title) for url, title in urls if not _is_blocked(url)]

    def priority_score(item):
        domain = _get_domain(item[0])
        return 0 if any(p in domain for p in PRIORITY_DOMAINS) else 1

    return sorted(filtered, key=priority_score)


class AnuInfrastructureScraper:
    def __init__(
        self,
        max_results: int = 3,
        timeout: float = 10.0,
        max_chars: int = 2000,
        extract_mode: str = "basic",
        max_retries: int = 2,
        backoff_factor: float = 1.5,
    ):
        self.max_results = max_results
        self.timeout = timeout
        self.max_chars = max_chars
        self.extract_mode = extract_mode
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        if extract_mode not in ["basic", "trafilatura"]:
            raise ValueError("extract_mode must be 'basic' or 'trafilatura'")

    async def get_facts(self, query: str):
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=self.max_results + 5))

        urls = [(r["href"], r["title"]) for r in results]

        urls = _sort_urls(urls)
        urls = urls[:self.max_results]

        if not urls:
            print("⚠️  All URLs were blocked domains, no results to scrape.")
            return []

        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
        async with httpx.AsyncClient(
            timeout=self.timeout,
            limits=limits,
            follow_redirects=True,
        ) as client:
            tasks = [
                self._fetch_with_retry(client, url, title) for url, title in urls
            ]
            docs = await asyncio.gather(*tasks, return_exceptions=True)

        slate_entries = []
        for d in docs:
            if isinstance(d, dict):
                slate_entries.append(d)
            elif isinstance(d, Exception):
                print(f"⚠️  Unhandled exception: {type(d).__name__}: {d}")

        return slate_entries

    async def _fetch_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        title: str,
    ):
        """Retry with exponential backoff on failure."""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                return await self._fetch_and_clean(client, url, title)
            except FETCH_ERRORS as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = self.backoff_factor ** attempt + random.uniform(0, 0.5)
                    print(
                        f"⚠️  Attempt {attempt + 1} failed [{url}]: "
                        f"{type(e).__name__} — retrying in {wait:.1f}s"
                    )
                    await asyncio.sleep(wait)
                else:
                    print(
                        f"❌ All {self.max_retries + 1} attempts failed [{url}]: "
                        f"{type(last_error).__name__}"
                    )
            except Exception as e:
                print(f"⚠️  Unexpected error [{url}]: {type(e).__name__}: {e}")
                return None

        return None

    async def _fetch_and_clean(
        self,
        client: httpx.AsyncClient,
        url: str,
        title: str,
    ):
        headers = random.choice(REALISTIC_HEADERS)

        await asyncio.sleep(random.uniform(0.1, 0.4))

        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        html = resp.text

        try:
            if self.extract_mode == "trafilatura":
                soup = BeautifulSoup(html, "html.parser")
                clean_text = trafilatura.extract(
                    html,
                    include_comments=False,
                    include_tables=True,
                    no_fallback=False,
                    favor_recall=True,
                ) or ""
                if len(clean_text.strip()) < 100:
                    print(
                        f"ℹ️  [{_get_domain(url)}] Using BS4 fallback"
                    )
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

        clean_text = clean_text[:self.max_chars]

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

    async def search(self, query: str) -> list[dict]:
        docs = await self.get_facts(query)
        results = []
        for i, d in enumerate(docs):
            results.append(
                {
                    "title": d["title"],
                    "url": d["source"],
                    "content": d["content"][:self.max_chars],
                    "raw_content": d["content"],
                    "score": 1.0 - i * 0.1,
                    "published_date": d.get("timestamp"),
                    "author": d.get("author"),
                }
            )
        return results

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
