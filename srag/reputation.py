import hashlib
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import pyarrow as pa
import lancedb

logger = logging.getLogger(__name__)

# ── Reputation schema ─────────────────────────────────────────────────────────

REPUTATION_SCHEMA = pa.schema([
    pa.field("record_id",            pa.string()),
    pa.field("domain",               pa.string()),
    pa.field("topic",                pa.string()),
    pa.field("retrieval_confidence", pa.float32()),
    pa.field("avg_chunk_quality",    pa.float32()),
    pa.field("useful_hit_rate",      pa.float32()),
    pa.field("irrelevance_rate",     pa.float32()),
    pa.field("failure_rate",         pa.float32()),
    pa.field("avg_latency",          pa.float32()),
    pa.field("total_scrapes",        pa.int32()),
    pa.field("last_updated",         pa.string()),
])

REPUTATION_TABLE  = "domain_reputation"
MIN_CONFIDENCE    = 0.1
EMA_WEIGHT        = 0.8
NEW_WEIGHT        = 0.2
DECAY_FACTOR      = 0.98
CONFIDENCE_FLOOR  = 0.1


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def _record_id(domain: str, topic: str) -> str:
    return hashlib.md5(f"{domain}:{topic}".encode()).hexdigest()[:16]


def _apply_decay(value: float, last_updated: str, now: datetime) -> float:
    """Apply time-based EMA decay proportional to days elapsed."""
    try:
        last = datetime.fromisoformat(last_updated)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        days   = (now - last).total_seconds() / 86400.0
        factor = DECAY_FACTOR ** days
        return max(CONFIDENCE_FLOOR, value * factor)
    except Exception:
        return value


def _compute_scrape_score(
    avg_chunk_quality: float,
    useful_hit_rate:   float,
    irrelevance_rate:  float,
    failure_rate:      float,
    avg_latency:       float,
) -> float:
    avg_chunk_quality = max(0.0, min(1.0, avg_chunk_quality))
    useful_hit_rate   = max(0.0, min(1.0, useful_hit_rate))
    irrelevance_rate  = max(0.0, min(1.0, irrelevance_rate))
    failure_rate      = max(0.0, min(1.0, failure_rate))
    avg_latency       = max(0.0, avg_latency)

    latency_score = 1.0 / (1.0 + avg_latency)

    score = (
        0.35 * avg_chunk_quality +
        0.25 * useful_hit_rate +
        0.15 * (1.0 - irrelevance_rate) +
        0.15 * (1.0 - failure_rate) +
        0.10 * latency_score
    )
    return max(0.0, min(1.0, score))


def _update_confidence(old: float, scrape_score: float) -> float:
    return max(MIN_CONFIDENCE, min(1.0, EMA_WEIGHT * old + NEW_WEIGHT * scrape_score))


# ── ReputationStore ───────────────────────────────────────────────────────────

class ReputationStore:
    """
    Persistent domain reputation system backed by LanceDB.
    Tracks per-domain, per-topic scrape quality over time.
    Calls LexiconStore.observe() after each successful update.
    """

    def __init__(
        self,
        db: lancedb.DBConnection,
        lexicon=None,    # LexiconStore | None — injected to avoid circular import
    ):
        self.db      = db
        self.lexicon = lexicon
        self._table  = self._init_table()

    def _init_table(self):
        if REPUTATION_TABLE not in self.db.table_names():
            return self.db.create_table(
                REPUTATION_TABLE,
                schema=REPUTATION_SCHEMA,
                mode="overwrite",
            )
        return self.db.open_table(REPUTATION_TABLE)

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, domain: str, topic: str) -> Optional[dict]:
        try:
            rid     = _record_id(domain, topic)
            results = (
                self._table.search()
                .where(f"record_id = '{rid}'")
                .limit(1)
                .to_list()
            )
            return results[0] if results else None
        except Exception:
            logger.exception("ReputationStore.get failed for %s/%s", domain, topic)
            return None

    def get_confidence(self, domain: str, topic: str) -> float:
        record = self.get(domain, topic)
        if record is None:
            return 0.5
        now   = datetime.now(timezone.utc)
        conf  = float(record.get("retrieval_confidence", 0.5))
        return _apply_decay(conf, record["last_updated"], now)

    def get_all(self) -> list[dict]:
        try:
            return self._table.to_pandas().to_dict(orient="records")
        except Exception:
            logger.exception("ReputationStore.get_all failed")
            return []

    # ── Write ─────────────────────────────────────────────────────────────────

    def update(
        self,
        domain:            str,
        topic:             str,
        query:             str,
        avg_chunk_quality: float,
        useful_hit_rate:   float,
        irrelevance_rate:  float,
        failure_rate:      float,
        avg_latency:       float,
    ) -> float:
        """
        Update reputation for a domain+topic pair after a scrape.
        Calls LexiconStore.observe() if confidence is high enough.
        Returns the new confidence score.
        """
        try:
            rid      = _record_id(domain, topic)
            existing = self.get(domain, topic)
            now      = datetime.now(timezone.utc)

            scrape_score = _compute_scrape_score(
                avg_chunk_quality=avg_chunk_quality,
                useful_hit_rate=useful_hit_rate,
                irrelevance_rate=irrelevance_rate,
                failure_rate=failure_rate,
                avg_latency=avg_latency,
            )

            if existing:
                old_conf      = _apply_decay(
                    float(existing.get("retrieval_confidence", 0.5)),
                    existing["last_updated"],
                    now,
                )
                total_scrapes = int(existing.get("total_scrapes", 0)) + 1
                n             = total_scrapes

                # EMA for all secondary metrics
                avg_chunk_quality = EMA_WEIGHT * float(existing.get("avg_chunk_quality", 0.0)) + NEW_WEIGHT * avg_chunk_quality
                useful_hit_rate   = EMA_WEIGHT * float(existing.get("useful_hit_rate",   0.0)) + NEW_WEIGHT * useful_hit_rate
                irrelevance_rate  = EMA_WEIGHT * float(existing.get("irrelevance_rate",  0.0)) + NEW_WEIGHT * irrelevance_rate
                failure_rate      = EMA_WEIGHT * float(existing.get("failure_rate",      0.0)) + NEW_WEIGHT * failure_rate
                avg_latency       = EMA_WEIGHT * float(existing.get("avg_latency",       0.0)) + NEW_WEIGHT * avg_latency

                new_confidence = _update_confidence(old_conf, scrape_score)
                self._table.delete(f"record_id = '{rid}'")
            else:
                new_confidence = _update_confidence(0.5, scrape_score)
                total_scrapes  = 1

            record = {
                "record_id":            rid,
                "domain":               domain,
                "topic":                topic,
                "retrieval_confidence": round(float(new_confidence), 4),
                "avg_chunk_quality":    round(float(avg_chunk_quality), 4),
                "useful_hit_rate":      round(float(useful_hit_rate), 4),
                "irrelevance_rate":     round(float(irrelevance_rate), 4),
                "failure_rate":         round(float(failure_rate), 4),
                "avg_latency":          round(float(avg_latency), 4),
                "total_scrapes":        total_scrapes,
                "last_updated":         now.isoformat(),
            }

            self._table.add([record])
            logger.debug(
                "Reputation updated: %s/%s confidence=%.3f scrape_score=%.3f",
                domain, topic, new_confidence, scrape_score,
            )

            # Notify lexicon — only on high confidence scrapes
            if self.lexicon is not None and new_confidence >= 0.65:
                try:
                    self.lexicon.observe(query=query, topic=topic, confidence=new_confidence)
                except Exception:
                    logger.exception("LexiconStore.observe failed silently")

            return new_confidence

        except Exception:
            logger.exception("ReputationStore.update failed for %s/%s", domain, topic)
            return 0.5

    # ── Export ────────────────────────────────────────────────────────────────

    def export_json(self, path: str) -> None:
        try:
            records = self.get_all()
            out     = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, default=str)
            logger.info("Reputation exported to %s (%d records)", path, len(records))
            print(f"📊 Reputation exported: {path} ({len(records)} records)")
        except Exception:
            logger.exception("ReputationStore.export_json failed")

    def summary(self, topic: Optional[str] = None) -> list[dict]:
        records = self.get_all()
        if topic:
            records = [r for r in records if r.get("topic") == topic]
        return sorted(
            records,
            key=lambda r: r.get("retrieval_confidence", 0.0),
            reverse=True,
        )


# ── ReputationAwareSelector ───────────────────────────────────────────────────

class ReputationAwareSelector:
    """
    Selects and orders URLs from a pool using reputation scores.

    Features:
    - Reputation-based sorting (primary signal)
    - Exploration budget (10% random selection for cold discovery)
    - Domain diversity limit (max N URLs per domain)
    - Cold start boost for unseen domains in priority list
    - Blocked domain hard filtering
    """

    def __init__(
        self,
        reputation:       ReputationStore,
        priority_domains: set[str],
        blocked_domains:  set[str],
        exploration_rate: float = 0.10,
        diversity_limit:  int   = 2,
        cold_start_boost: float = 0.05,
    ):
        self.reputation       = reputation
        self.priority_domains = priority_domains
        self.blocked_domains  = blocked_domains
        self.exploration_rate = exploration_rate
        self.diversity_limit  = diversity_limit
        self.cold_start_boost = cold_start_boost

    def select(
        self,
        urls:     list[tuple[str, str]],
        topic:    str,
        max_urls: int = 12,
    ) -> list[tuple[str, str]]:
        """
        Select and order URLs for fetching.

        Args:
            urls:     list of (url, title) from DuckDuckGo
            topic:    inferred topic for reputation lookup
            max_urls: hard cap on returned URLs

        Returns:
            Filtered, diversity-limited, reputation-sorted list.
        """
        # Hard filter blocked domains
        filtered = [
            (url, title) for url, title in urls
            if not self._is_blocked(url)
        ]

        if not filtered:
            return []

        # Split into exploitation pool and exploration budget
        n_explore  = max(1, int(len(filtered) * self.exploration_rate))
        n_exploit  = len(filtered) - n_explore

        # Score all URLs
        scored = [
            (url, title, self._score(url, topic))
            for url, title in filtered
        ]
        scored.sort(key=lambda x: x[2], reverse=True)

        exploit_pool  = scored[:n_exploit]
        explore_pool  = scored[n_exploit:]

        # Exploration — random sample from lower-scored URLs
        explore_sample = random.sample(
            explore_pool,
            min(n_explore, len(explore_pool))
        ) if explore_pool else []

        # Merge: exploitation first, then exploration
        combined = exploit_pool + explore_sample

        # Apply diversity limit — max N URLs per domain
        domain_counts: dict[str, int] = {}
        selected: list[tuple[str, str]] = []

        for url, title, score in combined:
            domain = _get_domain(url)
            count  = domain_counts.get(domain, 0)
            if count < self.diversity_limit:
                selected.append((url, title))
                domain_counts[domain] = count + 1
            if len(selected) >= max_urls:
                break

        logger.debug(
            "ReputationAwareSelector: %d → %d URLs (topic=%s, explored=%d)",
            len(urls), len(selected), topic, len(explore_sample),
        )

        return selected

    def _score(self, url: str, topic: str) -> float:
        domain     = _get_domain(url)
        confidence = self.reputation.get_confidence(domain, topic)

        # Cold start boost for unseen priority domains
        if confidence == 0.5 and any(p in domain for p in self.priority_domains):
            confidence += self.cold_start_boost

        return confidence

    def _is_blocked(self, url: str) -> bool:
        domain = _get_domain(url)
        return any(b in domain for b in self.blocked_domains)