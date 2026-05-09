import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Per-domain fetch record ───────────────────────────────────────────────────

@dataclass
class DomainFetchRecord:
    """
    Immutable record of a single fetch attempt for one URL.
    Created by AdaptiveFetcher, read by orchestrator.
    """
    domain:        str
    url:           str
    latency:       float        # seconds
    failed:        bool
    rate_limited:  bool         # 429 specifically
    content_empty: bool         # fetched but no useful content
    status_code:   Optional[int] = None


# ── Per-domain chunk record ───────────────────────────────────────────────────

@dataclass
class DomainChunkRecord:
    """
    Chunk quality stats for one domain after chunker + coherence filter.
    Created by orchestrator after index_documents() returns.
    """
    domain:         str
    chunks_total:   int
    chunks_kept:    int
    coherence_sum:  float       # sum of coherence scores of kept chunks

    @property
    def avg_coherence(self) -> float:
        if self.chunks_kept == 0:
            return 0.0
        return round(self.coherence_sum / self.chunks_kept, 4)

    @property
    def useful_hit_rate(self) -> float:
        if self.chunks_total == 0:
            return 0.0
        return round(self.chunks_kept / self.chunks_total, 4)

    @property
    def irrelevance_rate(self) -> float:
        return round(1.0 - self.useful_hit_rate, 4)


# ── Collector ─────────────────────────────────────────────────────────────────

@dataclass
class SideChannelCollector:
    """
    Stateless side-channel signal aggregator for one query lifecycle.

    Created fresh per query in orchestrator — no state leaks between queries.
    Populated during fetch (fetch records) and after indexing (chunk records).
    Read once by orchestrator._update_reputation() then discarded.

    Usage:
        collector = SideChannelCollector()
        docs = await scraper.get_facts(query, collector=collector)
        # ... chunker + indexer ...
        collector.record_chunks(domain, chunks_total, chunks_kept, coherence_sum)
        orchestrator._update_reputation(collector, topic, query)
    """

    fetch_records: list[DomainFetchRecord] = field(default_factory=list)
    chunk_records: list[DomainChunkRecord] = field(default_factory=list)
    _start_time:   float                   = field(default_factory=time.monotonic)

    # ── Record fetch ──────────────────────────────────────────────────────────

    def record_fetch(
        self,
        domain:        str,
        url:           str,
        latency:       float,
        failed:        bool,
        rate_limited:  bool        = False,
        content_empty: bool        = False,
        status_code:   Optional[int] = None,
    ) -> None:
        """
        Called by AdaptiveFetcher after each URL fetch attempt.
        Safe to call from async context — no locks needed (gather fills
        sequentially into collector from orchestrator after gather completes).
        """
        self.fetch_records.append(DomainFetchRecord(
            domain        = domain,
            url           = url,
            latency       = max(0.0, latency),
            failed        = failed,
            rate_limited  = rate_limited,
            content_empty = content_empty,
            status_code   = status_code,
        ))

    # ── Record chunks ─────────────────────────────────────────────────────────

    def record_chunks(
        self,
        domain:        str,
        chunks_total:  int,
        chunks_kept:   int,
        coherence_sum: float,
    ) -> None:
        """
        Called by orchestrator after index_documents() returns,
        once per domain in the session.
        """
        self.chunk_records.append(DomainChunkRecord(
            domain        = domain,
            chunks_total  = max(0, chunks_total),
            chunks_kept   = max(0, chunks_kept),
            coherence_sum = max(0.0, coherence_sum),
        ))

    # ── Aggregate per domain ──────────────────────────────────────────────────

    def aggregate(self) -> dict[str, dict]:
        """
        Aggregate all fetch + chunk records per domain into reputation signals.

        Returns:
            dict keyed by domain with all signals ready for ReputationStore.update():
            {
                "realpython.com": {
                    "avg_latency":       1.23,
                    "failure_rate":      0.0,
                    "useful_hit_rate":   0.75,
                    "irrelevance_rate":  0.25,
                    "avg_chunk_quality": 0.68,
                    "fetch_count":       2,
                    "chunk_total":       8,
                }
            }
        """
        # Group fetch records by domain
        fetch_by_domain: dict[str, list[DomainFetchRecord]] = {}
        for r in self.fetch_records:
            fetch_by_domain.setdefault(r.domain, []).append(r)

        # Group chunk records by domain
        chunk_by_domain: dict[str, DomainChunkRecord] = {}
        for r in self.chunk_records:
            chunk_by_domain[r.domain] = r

        result: dict[str, dict] = {}
        all_domains = set(fetch_by_domain) | set(chunk_by_domain)

        for domain in all_domains:
            fetches = fetch_by_domain.get(domain, [])
            chunks  = chunk_by_domain.get(domain)

            # Fetch signals
            if fetches:
                total_fetches  = len(fetches)
                failed         = sum(1 for f in fetches if f.failed)
                rate_limited   = sum(1 for f in fetches if f.rate_limited)
                latencies      = [f.latency for f in fetches if not f.failed]
                avg_latency    = sum(latencies) / len(latencies) if latencies else 0.0
                failure_rate   = failed / total_fetches
            else:
                total_fetches  = 0
                avg_latency    = 0.0
                failure_rate   = 0.0
                rate_limited   = 0

            # Chunk signals
            if chunks:
                avg_chunk_quality = chunks.avg_coherence
                useful_hit_rate   = chunks.useful_hit_rate
                irrelevance_rate  = chunks.irrelevance_rate
                chunk_total       = chunks.chunks_total
            else:
                avg_chunk_quality = 0.0
                useful_hit_rate   = 0.0
                irrelevance_rate  = 1.0
                chunk_total       = 0

            result[domain] = {
                "avg_latency":       round(avg_latency, 4),
                "failure_rate":      round(failure_rate, 4),
                "rate_limited":      rate_limited,
                "useful_hit_rate":   useful_hit_rate,
                "irrelevance_rate":  irrelevance_rate,
                "avg_chunk_quality": avg_chunk_quality,
                "fetch_count":       total_fetches,
                "chunk_total":       chunk_total,
            }

            logger.debug(
                "Collector.aggregate: %s | latency=%.2f failure=%.2f "
                "useful=%.2f quality=%.2f",
                domain, avg_latency, failure_rate,
                useful_hit_rate, avg_chunk_quality,
            )

        return result

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def elapsed(self) -> float:
        """Total wall time since collector was created."""
        return round(time.monotonic() - self._start_time, 3)

    def summary(self) -> dict:
        """Quick summary for logging."""
        agg = self.aggregate()
        return {
            "domains":       len(agg),
            "fetch_records": len(self.fetch_records),
            "chunk_records": len(self.chunk_records),
            "elapsed_s":     self.elapsed(),
            "domains_data":  agg,
        }

    def is_empty(self) -> bool:
        return not self.fetch_records and not self.chunk_records