import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional

import pyarrow as pa
import lancedb

logger = logging.getLogger(__name__)

# ── Schema ─────────────────────────────────────────────────────────────────
#
# Per-domain CSS selector overrides for DOMDistiller, plus a low-coherence
# streak counter used to flag domains where the global noise blocklist isn't
# enough — surfaced via `srag inspect --show domain-filters` for manual
# review, never auto-written. Same persistence shape as ReputationStore,
# just tracking selectors instead of trust scores.

DOMAIN_FILTER_SCHEMA = pa.schema([
    pa.field("record_id",            pa.string()),
    pa.field("domain",               pa.string()),
    pa.field("selectors",            pa.list_(pa.string())),
    pa.field("source",               pa.string()),   # "manual" | "default"
    pa.field("low_coherence_streak", pa.int32()),
    pa.field("total_scrapes",        pa.int32()),
    pa.field("flagged",              pa.bool_()),
    pa.field("last_updated",         pa.string()),
])

DOMAIN_FILTER_TABLE = "domain_filters"

# Consecutive low-coherence scrapes before a domain gets flagged for review.
FLAG_STREAK_THRESHOLD = 5
COHERENCE_FLAG_THRESHOLD = 0.3


def _record_id(domain: str) -> str:
    return hashlib.md5(domain.encode()).hexdigest()[:16]


class DomainFilterStore:
    """
    Persistent per-domain CSS selector overrides for DOMDistiller, backed by
    LanceDB — same pattern as ReputationStore.

    Starts empty for every domain (falls back to DOMDistiller's global
    DEFAULT_NOISE_SELECTORS). Populated two ways:

      1. Manual — set_selectors() / `srag inspect --show domain-filters`.
         You look at a domain that's leaking garbage past the global list
         and add a selector yourself.

      2. Flagged, never auto-written — record_coherence() tracks a rolling
         streak of low-coherence scrapes per domain. After
         FLAG_STREAK_THRESHOLD consecutive low-coherence scrapes, the domain
         is marked `flagged=True` and surfaces in `srag inspect`. This is
         NOT learnable from coherence scores alone — a low coherence score
         doesn't tell you *which* selector to add, only that the domain
         needs a human look. So flagging is the only automatic behavior;
         selectors themselves are always manual.
    """

    def __init__(self, db: lancedb.DBConnection):
        self.db = db
        self._table = self._init_table()

    def _init_table(self):
        if DOMAIN_FILTER_TABLE not in self.db.table_names():
            return self.db.create_table(
                DOMAIN_FILTER_TABLE,
                schema=DOMAIN_FILTER_SCHEMA,
                mode="overwrite",
            )
        return self.db.open_table(DOMAIN_FILTER_TABLE)

    # ── Read ──────────────────────────────────────────────────────────────

    def get(self, domain: str) -> Optional[dict]:
        try:
            rid = _record_id(domain)
            results = (
                self._table.search()
                .where(f"record_id = '{rid}'")
                .limit(1)
                .to_list()
            )
            return results[0] if results else None
        except Exception:
            logger.exception("DomainFilterStore.get failed for %s", domain)
            return None

    def get_selectors(self, domain: str) -> list[str]:
        """Manual override selectors for this domain. Empty if none set."""
        record = self.get(domain)
        if not record:
            return []
        return list(record.get("selectors") or [])

    def get_all(self) -> list[dict]:
        try:
            return self._table.to_pandas().to_dict(orient="records")
        except Exception:
            logger.exception("DomainFilterStore.get_all failed")
            return []

    def get_flagged(self) -> list[dict]:
        """Domains whose low-coherence streak crossed the threshold — needs a human look."""
        return [r for r in self.get_all() if r.get("flagged")]

    # ── Write ─────────────────────────────────────────────────────────────

    def set_selectors(self, domain: str, selectors: list[str]) -> None:
        """
        Manual write — add/replace CSS selector overrides for a domain.
        Also clears any existing flag/streak: if a domain was flagged because
        its global-blocklist coherence kept failing, a human has now acted on
        it, so the flag shouldn't linger in `srag inspect` output until the
        next scrape happens to confirm it. If the new selector doesn't
        actually fix the streak, record_coherence() will simply re-flag it
        within FLAG_STREAK_THRESHOLD scrapes either way.
        """
        try:
            rid = _record_id(domain)
            existing = self.get(domain)
            now = datetime.now(timezone.utc).isoformat()

            record = {
                "record_id":            rid,
                "domain":               domain,
                "selectors":            selectors,
                "source":               "manual",
                "low_coherence_streak": 0,
                "total_scrapes":        int(existing.get("total_scrapes", 0)) if existing else 0,
                "flagged":              False,
                "last_updated":         now,
            }

            if existing:
                self._table.delete(f"record_id = '{rid}'")
            self._table.add([record])
            logger.info("DomainFilterStore: %s selectors set (%d rules)", domain, len(selectors))

        except Exception:
            logger.exception("DomainFilterStore.set_selectors failed for %s", domain)

    def record_coherence(self, domain: str, avg_coherence: float) -> bool:
        """
        Call after every scrape with the document's average post-distillation
        coherence score. Tracks a rolling streak of low-coherence scrapes and
        flags the domain once FLAG_STREAK_THRESHOLD is hit.

        Returns True if this call caused the domain to become newly flagged
        (useful for logging — callers can log once instead of every scrape).
        """
        try:
            rid = _record_id(domain)
            existing = self.get(domain)
            now = datetime.now(timezone.utc).isoformat()
            is_low = avg_coherence < COHERENCE_FLAG_THRESHOLD

            if existing:
                streak = int(existing.get("low_coherence_streak", 0))
                streak = streak + 1 if is_low else 0
                total = int(existing.get("total_scrapes", 0)) + 1
                was_flagged = bool(existing.get("flagged", False))
                selectors = list(existing.get("selectors") or [])
                source = existing.get("source", "default")
            else:
                streak = 1 if is_low else 0
                total = 1
                was_flagged = False
                selectors = []
                source = "default"

            now_flagged = streak >= FLAG_STREAK_THRESHOLD

            record = {
                "record_id":            rid,
                "domain":               domain,
                "selectors":            selectors,
                "source":               source,
                "low_coherence_streak": streak,
                "total_scrapes":        total,
                "flagged":              now_flagged,
                "last_updated":         now,
            }

            if existing:
                self._table.delete(f"record_id = '{rid}'")
            self._table.add([record])

            newly_flagged = now_flagged and not was_flagged
            if newly_flagged:
                logger.warning(
                    "DomainFilterStore: %s flagged for review — %d consecutive low-coherence scrapes",
                    domain, streak,
                )
            return newly_flagged

        except Exception:
            logger.exception("DomainFilterStore.record_coherence failed for %s", domain)
            return False

    def clear_flag(self, domain: str) -> None:
        """Manual reset — e.g. after you've added selectors and confirmed it's fixed."""
        try:
            rid = _record_id(domain)
            existing = self.get(domain)
            if not existing:
                return
            self._table.delete(f"record_id = '{rid}'")
            existing["low_coherence_streak"] = 0
            existing["flagged"] = False
            existing["last_updated"] = datetime.now(timezone.utc).isoformat()
            self._table.add([existing])
        except Exception:
            logger.exception("DomainFilterStore.clear_flag failed for %s", domain)