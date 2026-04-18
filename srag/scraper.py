import asyncio
from datetime import datetime
import random
import hashlib  # add to existing imports at top

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
import trafilatura
from playwright.async_api import async_playwright


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
    "levelup.gitconnected.com",
    "datacamp.com",
    "youtube.com",
    "youtu.be",
    "en.wikipedia.org",
    "wikipedia.org",
    "pinterest.com",
    "goodreturns.in",
    "medium.com",
    "towardsdatascience.com",
    "betterexplained.com",
}

PRIORITY_DOMAINS = {
    "docs.python.org",
    "fastapi.tiangolo.com",
    "realpython.com",
    "stackoverflow.com",
    "docs.lancedb.com",
    "arxiv.org",
    "britannica.com",
    "sciencedirect.com",
    "nature.com",
    "scholar.google.com",
    "ncbi.nlm.nih.gov",
    "incometax.gov.in",
    "gst.gov.in",
    "mospi.gov.in",            # Ministry of Statistics
    "rbi.org.in",              # Reserve Bank of India 
    "pib.gov.in",              # Press Information Bureau
    "developers.google.com", 
    "developer.mozilla.org",   # MDN — web standards
    "huggingface.co",          # ML models/docs
    "pytorch.org",
    "tensorflow.org",
    "khanacademy.org",
    "mit.edu",
    "nptel.ac.in", 
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
    filtered = [(url, title) for url, title in urls if not _is_blocked(url)]

    def priority_score(item):
        domain = _get_domain(item[0])
        return 0 if any(p in domain for p in PRIORITY_DOMAINS) else 1

    return sorted(filtered, key=priority_score)


def _is_content_useful(text: str, min_chars: int = 200) -> bool:
    """Check if extracted text is actually useful content."""
    if len(text.strip()) < min_chars:
        return False
    words = text.split()
    if not words:
        return False
    short_words = sum(1 for w in words if len(w) <= 3)
    if short_words / len(words) > 0.6:
        return False
    return True


class AnuInfrastructureScraper:
    def __init__(
        self,
        max_results: int = 3,
        timeout: float = 10.0,
        max_chars: int = 2000,
        extract_mode: str = "trafilatura",
        max_retries: int = 2,
        backoff_factor: float = 1.5,
        use_playwright: bool = True,
        playwright_timeout: float = 15.0,
    ):
        self.max_results = max_results
        self.timeout = timeout
        self.max_chars = max_chars
        self.extract_mode = extract_mode
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.use_playwright = use_playwright
        self.playwright_timeout = playwright_timeout

        if extract_mode not in ["basic", "trafilatura"]:
            raise ValueError("extract_mode must be 'basic' or 'trafilatura'")

    def _expand_query(self, query: str) -> list[str]:
        """
        Generates semantically related search queries to broaden coverage.
        Returns original query + up to 2 expansions.
        """
        expansions = [query]

        # Expansion 1 — add current year for freshness
        if "2026" not in query and "2025" not in query:
            expansions.append(f"{query} 2026")

        # Expansion 2 — rephrase with common synonyms
        rephrase_map = {
            "tutorial": "guide",
            "guide": "tutorial",
            "how to": "how do I",
            "error": "issue fix",
            "fix": "solution",
            "rate": "percentage",
            "law": "regulation",
            "price": "cost",
            "best": "top",
        }
        rephrased = query
        for original, replacement in rephrase_map.items():
            if original in query.lower():
                rephrased = query.lower().replace(original, replacement)
                break
        if rephrased != query:
            expansions.append(rephrased)

        return expansions[:3]

    async def get_facts(self, query: str):
        # Expand query for broader coverage
        queries = self._expand_query(query)

        all_urls = []
        seen_urls = set()

        for q in queries:
            with DDGS() as ddgs:
                results = list(ddgs.text(q, max_results=self.max_results + 3))
            for r in results:
                url = r["href"]
                if url not in seen_urls and not _is_blocked(url):
                    seen_urls.add(url)
                    all_urls.append((url, r["title"]))

        # Sort by domain priority and cap
        all_urls = _sort_urls(all_urls)
        all_urls = all_urls[:self.max_results]

        if not all_urls:
            print("⚠️  All URLs were blocked domains.")
            return []

        print(f"ℹ️  Query expanded to {len(queries)} variants, {len(all_urls)} unique URLs")

        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
        async with httpx.AsyncClient(
            timeout=self.timeout,
            limits=limits,
            follow_redirects=True,
        ) as client:
            tasks = [
                self._fetch_with_retry(client, url, title)
                for url, title in all_urls
            ]
            docs = await asyncio.gather(*tasks, return_exceptions=True)

        seen_hashes = set()
        slate_entries = []
        for d in docs:
            if isinstance(d, Exception):
                if "429" in str(d):
                    print(f"⚠️  Rate limited on one of the fetches: {d}")
                else:
                    print(f"⚠️  Error during fetching: {type(d).__name__}: {d}")
                continue
            if not isinstance(d, dict):
                print(f"⚠️  Unhandled exception: {type(d).__name__}: {d}")
                continue

            # Check for duplicate content using hash
            content_snippet = d.get("content", "")[:500]  # Use first 500 chars for hashing
            content_hash = hashlib.md5(content_snippet.encode()).hexdigest()
            if content_hash in seen_hashes:
                print(f"⚠️  Duplicate content found for URL: {d.get('source', '')}")
                continue
            seen_hashes.add(content_hash)

            slate_entries.append(d)

        return slate_entries

    async def search(self, query: str) -> list[dict]:
        """Tavily-style search API."""
        docs = await self.get_facts(query)
        results = []
        for i, d in enumerate(docs):
            results.append({
                "title": d["title"],
                "url": d["source"],
                "content": d["content"][:self.max_chars],
                "raw_content": d["content"],
                "score": 1.0 - i * 0.1,
                "published_date": d.get("timestamp"),
                "author": d.get("author"),
            })
        return results

    async def _fetch_with_retry(self, client, url, title):
        """Retry with exponential backoff, Playwright as final fallback."""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                return await self._fetch_and_clean(client, url, title)
            except FETCH_ERRORS as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = self.backoff_factor ** attempt + random.uniform(0, 0.5)
                    print(f"⚠️  Attempt {attempt + 1} failed [{url}]: {type(e).__name__} — retrying in {wait:.1f}s")
                    await asyncio.sleep(wait)
                else:
                    if self.use_playwright:
                        print(f"🎭 Falling back to Playwright for: {url}")
                        return await self._playwright_fetch(url, title)
                    print(f"❌ All attempts failed [{url}]: {type(last_error).__name__}")
            except Exception as e:
                print(f"⚠️  Unexpected error [{url}]: {type(e).__name__}: {e}")
                return None

        return None

    async def _playwright_fetch(self, url: str, title: str):
        """
        Headless browser fallback for JS-heavy pages.
        Only fires when httpx + trafilatura + BS4 all fail.
        """
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-dev-shm-usage"],
                )
                context = await browser.new_context(
                    user_agent=random.choice(REALISTIC_HEADERS)["User-Agent"],
                    java_script_enabled=True,
                )
                page = await context.new_page()

                # Block images, fonts, media — only need HTML content
                await page.route(
                    "**/*.{png,jpg,jpeg,gif,svg,ico,woff,woff2,ttf,mp4,mp3}",
                    lambda route: route.abort()
                )

                await page.goto(
                    url,
                    timeout=int(self.playwright_timeout * 1000),
                    wait_until="domcontentloaded"
                )

                # Wait for main content to load
                await page.wait_for_timeout(1500)

                html = await page.content()
                await browser.close()

            # Extract from fully rendered HTML
            soup = BeautifulSoup(html, "html.parser")
            clean_text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
                favor_recall=True,
            ) or ""

            if not _is_content_useful(clean_text):
                for el in soup(["nav", "footer", "script", "style", "header"]):
                    el.decompose()
                clean_text = soup.get_text(separator=" ", strip=True)

            clean_text = clean_text[:self.max_chars]

            if not clean_text.strip():
                print(f"⚠️  Playwright also got empty content: {url}")
                return None

            print(f"✅ Playwright succeeded: {url}")

            return {
                "source": url,
                "title": self._extract_title(soup) or title,
                "content": clean_text,
                "timestamp": self._extract_date(soup) or datetime.utcnow().isoformat(),
                "author": self._extract_author(soup),
                "image": self._extract_image(soup),
            }

        except Exception as e:
            print(f"❌ Playwright failed [{url}]: {type(e).__name__}: {e}")
            return None

    async def _fetch_and_clean(self, client, url, title):
        """Primary fetch via httpx with trafilatura/BS4 extraction."""
        headers = random.choice(REALISTIC_HEADERS)
        await asyncio.sleep(random.uniform(0.1, 0.4))

        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        html = resp.text

        try:
            soup = BeautifulSoup(html, "html.parser")

            if self.extract_mode == "trafilatura":
                clean_text = trafilatura.extract(
                    html,
                    include_comments=False,
                    include_tables=True,
                    no_fallback=False,
                    favor_recall=True,
                ) or ""

                if not _is_content_useful(clean_text):
                    print(f"ℹ️  [{_get_domain(url)}] Using BS4 fallback")
                    for el in soup(["nav", "footer", "script", "style", "header"]):
                        el.decompose()
                    clean_text = soup.get_text(separator=" ", strip=True)
            else:
                for el in soup(["nav", "footer", "script", "style", "header"]):
                    el.decompose()
                clean_text = soup.get_text(separator=" ", strip=True)

        except Exception as e:
            print(f"⚠️  Parsing failed [{url}]: {e}")
            return None

        clean_text = clean_text[:self.max_chars]

        if not clean_text.strip():
            print(f"⚠️  Empty content, skipping: {url}")
            return None

        return {
            "source": url,
            "title": self._extract_title(soup) or title,
            "content": clean_text,
            "timestamp": self._extract_date(soup) or datetime.utcnow().isoformat(),
            "author": self._extract_author(soup),
            "image": self._extract_image(soup),
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