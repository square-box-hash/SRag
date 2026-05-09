import asyncio
import hashlib
import logging
import random
import time
from datetime import datetime
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
import trafilatura
from playwright.async_api import async_playwright

from srag.collector import SideChannelCollector
from srag.adaptive_concurrency import ConcurrencyController, ManagedSlot

logger = logging.getLogger(__name__)

# ── Timeouts ──────────────────────────────────────────────────────────────────

TOTAL_SESSION_TIMEOUT = 45.0
PER_URL_TIMEOUT = 10.0
PLAYWRIGHT_TIMEOUT = 15.0

# Global semaphore: limit concurrent Playwright browser instances
_PLAYWRIGHT_SEMAPHORE = asyncio.Semaphore(2)

# ── Headers / domains ─────────────────────────────────────────────────────────

REALISTIC_HEADERS = [
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "identity",
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
        "Accept-Encoding": "identity",
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
        "Accept-Encoding": "identity",
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
    "suzukacircuit.jp",
    "motorsporttickets.com",
    "f1experiences.com",
    "plainenglish.io",
    "pub.towardsai.net",
    "towardsai.net",
    "informalnewz.com",
    "formulaonehistory.com",
    "bestcalendarprintable",
    "motorbiscuit.com",
    "f1dailybrief.com",
    "gpfans.com",
    "formulaonehistory.com",
    "f1-fansite.com",
    "motorsportstats.com",
    "total-motorsport.com",
}

PRIORITY_DOMAINS = {
    "docs.python.org",
    "fastapi.tiangolo.com",
    "realpython.com",
    "docs.lancedb.com",
    "arxiv.org",
    "stackoverflow.com",
    "britannica.com",
    "sciencedirect.com",
    "nature.com",
    "ncbi.nlm.nih.gov",
    "incometax.gov.in",
    "gst.gov.in",
    "mospi.gov.in",
    "rbi.org.in",
    "pib.gov.in",
    "developers.google.com",
    "developer.mozilla.org",
    "huggingface.co",
    "pytorch.org",
    "tensorflow.org",
    "khanacademy.org",
    "mit.edu",
    "nptel.ac.in",
    "formula1.com",
    "espncricinfo.com",
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


def _is_content_useful(text: str, min_chars: int = 200) -> bool:
    if len(text.strip()) < min_chars:
        return False
    words = text.split()
    if not words:
        return False
    short_words = sum(1 for w in words if len(w) <= 3)
    if short_words / len(words) > 0.6:
        return False
    return True


# ── Scraper ───────────────────────────────────────────────────────────────────


class AnuInfrastructureScraper:
    def __init__(
        self,
        max_results: int = 3,
        timeout: float = PER_URL_TIMEOUT,
        max_chars: int = 2000,
        extract_mode: str = "trafilatura",
        max_retries: int = 2,
        backoff_factor: float = 1.5,
        use_playwright: bool = True,
        playwright_timeout: float = PLAYWRIGHT_TIMEOUT,
        concurrency_controller: Optional[ConcurrencyController] = None,
        query_intelligence=None,  # QueryIntelligence | None
        topic_classifier=None,  # TopicClassifier | None
        reputation_selector=None,  # ReputationAwareSelector | None
    ):
        self.max_results = max_results
        self.timeout = timeout
        self.max_chars = max_chars
        self.extract_mode = extract_mode
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.use_playwright = use_playwright
        self.playwright_timeout = playwright_timeout
        self.controller = concurrency_controller or ConcurrencyController()
        self.query_intelligence = query_intelligence
        self.topic_classifier = topic_classifier
        self.reputation_selector = reputation_selector

        if extract_mode not in ["basic", "trafilatura"]:
            raise ValueError("extract_mode must be 'basic' or 'trafilatura'")

    async def get_facts(
        self,
        query: str,
        collector: Optional[SideChannelCollector] = None,
    ) -> list[dict]:
        """
        Full scrape pipeline with timeout + cancellation control.
        Returns partial results if session deadline is hit.
        """
        # ── Topic + query intelligence ────────────────────────────────────────
        topic = "general"
        ambiguous = False

        if self.topic_classifier:
            topic_result = self.topic_classifier.predict(query)
            topic = topic_result.primary
            ambiguous = topic_result.ambiguous
            logger.debug(
                "Topic: %s (%.2f) ambiguous=%s",
                topic,
                topic_result.confidence,
                ambiguous,
            )

        # ── Query variants ────────────────────────────────────────────────────
        if self.query_intelligence:
            plan = self.query_intelligence.rewrite(query, topic, ambiguous)
            queries = plan.get_queries()
            logger.debug("Query plan: %s", plan.summary())
        else:
            queries = [query]

        # ── URL discovery ─────────────────────────────────────────────────────
        all_urls: list[tuple[str, str]] = []
        seen_urls: set[str] = set()

        for q in queries:
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(q, max_results=self.max_results + 3))
                for r in results:
                    url = r["href"]
                    if url not in seen_urls:
                        seen_urls.add(url)
                        all_urls.append((url, r["title"]))
            except Exception as e:
                logger.warning("DDGS search failed for query '%s': %s", q, e)
                continue

        if not all_urls:
            logger.warning("No URLs found for query: %s", query)
            return []

        # ── URL selection (reputation-aware) ──────────────────────────────────
        if self.reputation_selector:
            all_urls = self.reputation_selector.select(
                urls=all_urls,
                topic=topic,
                max_urls=self.max_results,
            )
        else:
            # Fallback — basic domain filter + cap
            all_urls = [
                (url, title)
                for url, title in all_urls
                if not any(b in _get_domain(url) for b in BLOCKED_DOMAINS)
            ][: self.max_results]

        if not all_urls:
            logger.warning("All URLs filtered out for query: %s", query)
            return []

        logger.info(
            "Query expanded to %d variants, %d unique URLs (topic=%s)",
            len(queries),
            len(all_urls),
            topic,
        )

        # ── Fetch with session timeout ────────────────────────────────────────
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)

        async with httpx.AsyncClient(
            timeout=self.timeout,
            limits=limits,
            follow_redirects=True,
        ) as client:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.create_task(
                    self._fetch_with_retry(client, url, title, collector)
                )
                for url, title in all_urls
            ]

            try:
                # Session-level deadline — return partial results if hit
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=TOTAL_SESSION_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Session timeout (%.1fs) hit — returning partial results",
                    TOTAL_SESSION_TIMEOUT,
                )

                for task in tasks:
                    if not task.done():
                        task.cancel()
                # Cancel remaining tasks and collect completed ones
                results = []
                for task in tasks:
                    if task.done() and not task.cancelled():
                        try:
                            results.append(task.result())
                        except Exception as e:
                            logger.warning("Task failed: %s", e)
                            results.append(e)

        # ── Log results ────────────────────────────────────────────────────────
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Fetch error: %s", r)
            elif r is None:
                logger.info("No content fetched for one URL.")
            else:
                logger.info(
                    "Fetched content from %s (length=%d)",
                    r["source"],
                    len(r["content"]),
                )

        # ── Dedup + filter ────────────────────────────────────────────────────
        seen_hashes: set[str] = set()
        slate_entries: list[dict] = []

        for d in results:
            if isinstance(d, Exception):
                if "429" in str(d):
                    logger.warning("Rate limited: %s", d)
                else:
                    logger.warning("Unhandled exception: %s: %s", type(d).__name__, d)
                continue
            if not isinstance(d, dict):
                continue

            content_hash = hashlib.md5(
                d.get("content", "")[:500].encode()
            ).hexdigest()

            if content_hash in seen_hashes:
                logger.info("Duplicate content skipped: %s", d.get("source", ""))
                continue

            seen_hashes.add(content_hash)
            slate_entries.append(d)

        return slate_entries

    async def search(self, query: str) -> list[dict]:
        """Tavily-style search API — routes through full pipeline."""
        collector = SideChannelCollector()
        docs = await self.get_facts(query, collector=collector)
        results: list[dict] = []
        for i, d in enumerate(docs, start=1):
            results.append(
                {
                    "title": d["title"],
                    "url": d["source"],
                    "content": d["content"][: self.max_chars],
                    "raw_content": d["content"],
                    "score": 1.0 - i * 0.1,
                    "published_date": d.get("timestamp"),
                    "author": d.get("author"),
                    "domain": _get_domain(d["source"]),
                    "topic": d.get("topic", "general"),
                }
            )
        return results

    async def _fetch_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        title: str,
        collector: Optional[SideChannelCollector] = None,
    ) -> Optional[dict]:
        """Retry with exponential backoff + cancellation handling."""
        domain = _get_domain(url)
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            start = time.monotonic()
            try:
                async with ManagedSlot(self.controller, domain) as token:
                    result = await self._fetch_and_clean(client, url, title)
                    latency = token.elapsed()

                    self.controller.record(token, latency=latency, failed=False)

                    if collector:
                        collector.record_fetch(
                            domain=domain,
                            url=url,
                            latency=latency,
                            failed=False,
                            content_empty=result is None,
                        )
                    return result

            except asyncio.CancelledError:
                logger.debug("Fetch cancelled: %s", url)
                if collector:
                    collector.record_fetch(
                        domain=domain,
                        url=url,
                        latency=time.monotonic() - start,
                        failed=True,
                    )
                # propagate cancellation — don't swallow
                raise

            except FETCH_ERRORS as e:
                latency = time.monotonic() - start
                last_error = e
                rate_limited = "429" in str(e)

                if collector:
                    collector.record_fetch(
                        domain=domain,
                        url=url,
                        latency=latency,
                        failed=True,
                        rate_limited=rate_limited,
                        status_code=getattr(
                            getattr(e, "response", None), "status_code", None
                        ),
                    )

                # synthetic token for recording failures on domain
                self.controller.record(
                    token=type(
                        "T",
                        (),
                        {"domain": domain, "acquired_at": start, "effective": 1},
                    )(),
                    latency=latency,
                    failed=True,
                    rate_limited=rate_limited,
                )

                if attempt < self.max_retries:
                    wait = self.backoff_factor**attempt + random.uniform(0, 0.5)
                    logger.warning(
                        "Attempt %d failed [%s]: %s — retrying in %.1fs",
                        attempt + 1,
                        url,
                        type(e).__name__,
                        wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    if self.use_playwright:
                        logger.info("Falling back to Playwright: %s", url)
                        return await self._playwright_fetch(url, title, collector)
                    logger.error(
                        "All attempts failed [%s]: %s",
                        url,
                        type(last_error).__name__ if last_error else "Unknown",
                    )

            except Exception as e:
                logger.warning(
                    "Unexpected error [%s]: %s: %s", url, type(e).__name__, e
                )
                if collector:
                    collector.record_fetch(
                        domain=domain,
                        url=url,
                        latency=time.monotonic() - start,
                        failed=True,
                    )
                return None

        return None

    async def _playwright_fetch(
        self,
        url: str,
        title: str,
        collector: Optional[SideChannelCollector] = None,
    ) -> Optional[dict]:
        domain = _get_domain(url)
        start = time.monotonic()

        try:
            async with _PLAYWRIGHT_SEMAPHORE:
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
                    await page.route(
                        "**/*.{png,jpg,jpeg,gif,svg,ico,woff,woff2,ttf,mp4,mp3}",
                        lambda route: route.abort(),
                    )
                    await page.goto(
                        url,
                        timeout=int(self.playwright_timeout * 1000),
                        wait_until="domcontentloaded",
                    )
                    await page.wait_for_timeout(1500)
                    html = await page.content()
                    await browser.close()

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

            clean_text = clean_text[: self.max_chars]
            latency = time.monotonic() - start

            if not clean_text.strip():
                logger.warning("Playwright got empty content: %s", url)
                if collector:
                    collector.record_fetch(
                        domain=domain,
                        url=url,
                        latency=latency,
                        failed=False,
                        content_empty=True,
                    )
                return None

            logger.info("Playwright succeeded: %s", url)
            if collector:
                collector.record_fetch(
                    domain=domain,
                    url=url,
                    latency=latency,
                    failed=False,
                )

            return {
                "source": url,
                "title": self._extract_title(soup) or title,
                "content": clean_text,
                "timestamp": self._extract_date(soup)
                or datetime.utcnow().isoformat(),
                "author": self._extract_author(soup),
                "image": self._extract_image(soup),
            }

        except asyncio.CancelledError:
            logger.debug("Playwright fetch cancelled: %s", url)
            raise
        except Exception as e:
            logger.error("Playwright failed [%s]: %s: %s", url, type(e).__name__, e)
            if collector:
                collector.record_fetch(
                    domain=domain,
                    url=url,
                    latency=time.monotonic() - start,
                    failed=True,
                )
            return None

    async def _fetch_and_clean(
        self,
        client: httpx.AsyncClient,
        url: str,
        title: str,
    ) -> Optional[dict]:
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
                    logger.info("[%s] Using BS4 fallback", _get_domain(url))
                    for el in soup(
                        ["nav", "footer", "script", "style", "header"]
                    ):
                        el.decompose()
                    clean_text = soup.get_text(separator=" ", strip=True)
            else:
                for el in soup(["nav", "footer", "script", "style", "header"]):
                    el.decompose()
                clean_text = soup.get_text(separator=" ", strip=True)

        except Exception as e:
            logger.warning("Parsing failed [%s]: %s", url, e)
            return None

        clean_text = clean_text[: self.max_chars]

        if not clean_text.strip():
            logger.debug("Empty content, skipping: %s", url)
            return None

        return {
            "source": url,
            "title": self._extract_title(soup) or title,
            "content": clean_text,
            "timestamp": self._extract_date(soup)
            or datetime.utcnow().isoformat(),
            "author": self._extract_author(soup),
            "image": self._extract_image(soup),
        }

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            return h1.get_text(strip=True)
        return None

    def _extract_date(self, soup: BeautifulSoup) -> Optional[str]:
        meta = soup.find(
            "meta", {"property": "article:published_time"}
        ) or soup.find("meta", {"name": "date"})
        if meta and meta.get("content"):
            return meta["content"]
        return None

    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        meta = soup.find("meta", {"name": "author"})
        if meta and meta.get("content"):
            return meta["content"]
        return None

    def _extract_image(self, soup: BeautifulSoup) -> Optional[str]:
        meta = soup.find("meta", {"property": "og:image"})
        if meta and meta.get("content"):
            return meta["content"]
        return None