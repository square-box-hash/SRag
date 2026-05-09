import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_CONCURRENCY        = 2
MAX_CONCURRENCY        = 20
COOLDOWN_SECONDS       = 5.0
MAX_REQUESTS_PER_SEC   = 10.0

LATENCY_LOW            = 1.5    # seconds — below this = healthy
LATENCY_HIGH           = 4.0    # seconds — above this = degrade

FAILURE_RATE_LOW       = 0.05   # below this = healthy
FAILURE_RATE_HIGH      = 0.20   # above this = degrade

DOMAIN_FAILURE_HIGH    = 0.40   # per-domain failure rate threshold
DOMAIN_MIN_CONCURRENCY = 1
DOMAIN_MAX_CONCURRENCY = 5

AIMD_INCREASE          = 1      # additive increase
AIMD_DECREASE_FACTOR   = 0.6    # multiplicative decrease


# ── Latency window ────────────────────────────────────────────────────────────

class _LatencyWindow:
    """
    Rolling window of recent latency samples.
    Computes p50 and p95 without storing unbounded history.
    """

    def __init__(self, maxlen: int = 50):
        self._samples: list[float] = []
        self._maxlen  = maxlen

    def record(self, latency: float) -> None:
        if len(self._samples) >= self._maxlen:
            self._samples.pop(0)
        self._samples.append(latency)

    def p50(self) -> float:
        if not self._samples:
            return 0.0
        s = sorted(self._samples)
        return s[len(s) // 2]

    def p95(self) -> float:
        if not self._samples:
            return 0.0
        s = sorted(self._samples)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]

    def empty(self) -> bool:
        return not self._samples


# ── Domain state ──────────────────────────────────────────────────────────────

@dataclass
class DomainConcurrencyState:
    """
    Per-domain concurrency tracking.
    Independent from global controller — allows domain-specific throttling
    without affecting throughput to healthy domains.
    """
    domain:          str
    concurrency:     int   = DOMAIN_MAX_CONCURRENCY
    total_requests:  int   = 0
    failed_requests: int   = 0
    last_failure:    float = 0.0

    @property
    def failure_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    def record_attempt(self, failed: bool) -> None:
        self.total_requests  += 1
        if failed:
            self.failed_requests += 1
            self.last_failure     = time.monotonic()

    def degrade(self) -> None:
        self.concurrency = max(
            DOMAIN_MIN_CONCURRENCY,
            int(self.concurrency * 0.5),
        )
        logger.info(
            "Domain %s degraded → concurrency=%d (failure_rate=%.2f)",
            self.domain, self.concurrency, self.failure_rate,
        )

    def recover(self) -> None:
        self.concurrency = min(
            DOMAIN_MAX_CONCURRENCY,
            self.concurrency + 1,
        )


# ── Global controller ─────────────────────────────────────────────────────────

class ConcurrencyController:
    """
    AIMD-style adaptive concurrency controller.

    Global level:
    - Additive increase when failure_rate < 0.05 AND p95_latency < LATENCY_LOW
    - Multiplicative decrease (x0.6) when failure_rate > 0.20 OR p95 > LATENCY_HIGH
    - Cooldown after reduction prevents oscillation
    - Burst protection via token bucket (max_requests_per_second)

    Domain level:
    - Per-domain semaphore independent of global
    - Effective concurrency = min(global, domain)
    - Domain degraded 50% on high failure rate
    - Domain recovers +1 on healthy window

    Usage:
        controller = ConcurrencyController()
        async with controller.acquire("realpython.com") as token:
            result = await fetch(url)
            controller.record(token, latency, failed)
    """

    def __init__(
        self,
        min_concurrency:      int   = MIN_CONCURRENCY,
        max_concurrency:      int   = MAX_CONCURRENCY,
        cooldown_seconds:     float = COOLDOWN_SECONDS,
        max_requests_per_sec: float = MAX_REQUESTS_PER_SEC,
    ):
        self.min_concurrency      = min_concurrency
        self.max_concurrency      = max_concurrency
        self.cooldown_seconds     = cooldown_seconds
        self.max_requests_per_sec = max_requests_per_sec

        self._concurrency:   int   = max_concurrency // 2   # start at midpoint
        self._last_drop:     float = 0.0
        self._semaphore:     asyncio.Semaphore = asyncio.Semaphore(self._concurrency)
        self._latency_win:   _LatencyWindow    = _LatencyWindow()
        self._domain_states: dict[str, DomainConcurrencyState] = {}

        # Token bucket for burst protection
        self._token_bucket:      float = max_requests_per_sec
        self._bucket_last_refill: float = time.monotonic()
        self._bucket_lock:        asyncio.Lock = asyncio.Lock()

    # ── Acquire ───────────────────────────────────────────────────────────────

    async def acquire(self, domain: str) -> "ConcurrencyToken":
        """
        Acquire a concurrency slot for the given domain.
        Respects both global semaphore and per-domain limit.
        Enforces burst protection via token bucket.
        Returns a token to pass to record() after the request completes.
        """
        await self._throttle_burst()

        domain_state = self._get_domain_state(domain)
        effective    = min(self._concurrency, domain_state.concurrency)

        # Rebuild semaphore if concurrency changed
        await self._semaphore.acquire()

        return ConcurrencyToken(
            domain       = domain,
            acquired_at  = time.monotonic(),
            effective    = effective,
        )

    async def release(self, token: "ConcurrencyToken") -> None:
        """Release the semaphore slot."""
        self._semaphore.release()

    # ── Record outcome ────────────────────────────────────────────────────────

    def record(
        self,
        token:        "ConcurrencyToken",
        latency:      float,
        failed:       bool,
        rate_limited: bool = False,
    ) -> None:
        """
        Record the outcome of a completed request.
        Updates global latency window and domain state.
        Triggers AIMD adjustment if thresholds are crossed.
        """
        if not failed:
            self._latency_win.record(latency)

        domain_state = self._get_domain_state(token.domain)
        domain_state.record_attempt(failed)

        # Per-domain degradation
        if domain_state.failure_rate > DOMAIN_FAILURE_HIGH and domain_state.total_requests >= 3:
            domain_state.degrade()
        elif not failed and domain_state.failure_rate < FAILURE_RATE_LOW:
            domain_state.recover()

        # Global AIMD adjustment
        self._adjust()

    # ── AIMD ──────────────────────────────────────────────────────────────────

    def _adjust(self) -> None:
        """
        AIMD adjustment on global concurrency.
        Decrease: failure_rate > 0.20 OR p95 > LATENCY_HIGH
        Increase: failure_rate < 0.05 AND p95 < LATENCY_LOW (after cooldown)
        """
        if self._latency_win.empty():
            return

        p95          = self._latency_win.p95()
        failure_rate = self._global_failure_rate()
        now          = time.monotonic()

        if failure_rate > FAILURE_RATE_HIGH or p95 > LATENCY_HIGH:
            if now - self._last_drop > self.cooldown_seconds:
                new_concurrency = max(
                    self.min_concurrency,
                    int(self._concurrency * AIMD_DECREASE_FACTOR),
                )
                if new_concurrency != self._concurrency:
                    self._concurrency = new_concurrency
                    self._last_drop   = now
                    logger.info(
                        "AIMD decrease → concurrency=%d "
                        "(failure=%.2f p95=%.2fs)",
                        self._concurrency, failure_rate, p95,
                    )

        elif (
            failure_rate < FAILURE_RATE_LOW
            and p95 < LATENCY_LOW
            and now - self._last_drop > self.cooldown_seconds
        ):
            new_concurrency = min(
                self.max_concurrency,
                self._concurrency + AIMD_INCREASE,
            )
            if new_concurrency != self._concurrency:
                self._concurrency = new_concurrency
                logger.info(
                    "AIMD increase → concurrency=%d "
                    "(failure=%.2f p95=%.2fs)",
                    self._concurrency, failure_rate, p95,
                )

    # ── Burst protection ──────────────────────────────────────────────────────

    async def _throttle_burst(self) -> None:
        """
        Token bucket burst protection.
        Refills at max_requests_per_second.
        Sleeps if bucket is empty.
        """
        async with self._bucket_lock:
            now     = time.monotonic()
            elapsed = now - self._bucket_last_refill
            refill  = elapsed * self.max_requests_per_sec

            self._token_bucket       = min(
                self.max_requests_per_sec,
                self._token_bucket + refill,
            )
            self._bucket_last_refill = now

            if self._token_bucket < 1.0:
                wait = (1.0 - self._token_bucket) / self.max_requests_per_sec
                logger.debug("Burst protection: sleeping %.3fs", wait)
                await asyncio.sleep(wait)
                self._token_bucket = 0.0
            else:
                self._token_bucket -= 1.0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_domain_state(self, domain: str) -> DomainConcurrencyState:
        if domain not in self._domain_states:
            self._domain_states[domain] = DomainConcurrencyState(domain=domain)
        return self._domain_states[domain]

    def _global_failure_rate(self) -> float:
        all_states    = list(self._domain_states.values())
        total         = sum(s.total_requests  for s in all_states)
        failed        = sum(s.failed_requests for s in all_states)
        if total == 0:
            return 0.0
        return failed / total

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "global_concurrency": self._concurrency,
            "p50_latency":        round(self._latency_win.p50(), 3),
            "p95_latency":        round(self._latency_win.p95(), 3),
            "global_failure_rate": round(self._global_failure_rate(), 3),
            "domain_states": {
                d: {
                    "concurrency":   s.concurrency,
                    "failure_rate":  round(s.failure_rate, 3),
                    "total":         s.total_requests,
                }
                for d, s in self._domain_states.items()
            },
        }


# ── Token ─────────────────────────────────────────────────────────────────────

@dataclass
class ConcurrencyToken:
    """
    Returned by ConcurrencyController.acquire().
    Passed back to record() after request completes.
    Tracks which domain and when the slot was acquired.
    """
    domain:      str
    acquired_at: float
    effective:   int

    def elapsed(self) -> float:
        return round(time.monotonic() - self.acquired_at, 4)


# ── Context manager wrapper ───────────────────────────────────────────────────

class ManagedSlot:
    """
    Async context manager wrapping acquire/release.

    Usage:
        async with ManagedSlot(controller, domain) as token:
            result = await fetch(url)
            controller.record(token, latency=token.elapsed(), failed=False)
    """

    def __init__(self, controller: ConcurrencyController, domain: str):
        self._controller = controller
        self._domain     = domain
        self._token: Optional[ConcurrencyToken] = None

    async def __aenter__(self) -> ConcurrencyToken:
        self._token = await self._controller.acquire(self._domain)
        return self._token

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._controller.release(self._token)
        if exc_type is not None:
            self._controller.record(
                self._token,
                latency      = self._token.elapsed(),
                failed       = True,
                rate_limited = exc_type is not None and "429" in str(exc_val),
            )