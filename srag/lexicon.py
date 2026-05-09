import hashlib
import logging
from datetime import datetime, timezone
from math import log
from typing import Any, Dict, List, Optional


import pyarrow as pa
import lancedb
import random


logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────


LEXICON_TABLE = "lexicon"


GRADUATION_THRESHOLD     = 0.70   # avg_confidence to graduate candidate → active
GRADUATION_OBSERVATIONS  = 8      # min observations to graduate
DEMOTION_CONSECUTIVE     = 5      # active → candidate
SUPPRESSION_FROM_CAND    = 10     # candidate → suppressed
SUPPRESSION_FROM_ACTIVE  = 13     # active → suppressed (midpoint of 12–15)
LOW_CONFIDENCE_THRESHOLD = 0.50   # below this = "low score" observation
DECAY_FACTOR             = 0.98   # per day
AVG_CONFIDENCE_FLOOR     = 0.20   # old terms never fully vanish
MAX_LEXICON_TERMS        = 3      # max terms injected into query expansion
MIN_TERM_WEIGHT          = 0.30   # minimum weight to be included in expansion
MIN_OBS_FULL_WEIGHT      = 10     # observations needed for full obs_factor


# Probation / exploration
SUPPRESSED_PROBATION_DAYS = 30    # days before suppressed term eligible for recovery
EXPLORATION_PROBABILITY   = 0.02  # 2% chance of exploring a suppressed term


STOP_WORDS = {
    "the", "and", "for", "with", "this", "that", "from", "into", "about",
    "what", "when", "where", "which", "how", "why", "who", "are", "was",
    "were", "has", "have", "had", "will", "would", "could", "should",
    "can", "may", "might", "shall", "been", "being", "its", "than",
    "then", "they", "their", "them", "these", "those", "such", "also",
    "but", "not", "yet", "nor", "any", "all", "each", "both", "more",
    "most", "other", "some", "same", "just", "like", "very", "too",
}


# ── Schema ────────────────────────────────────────────────────────────────────


LEXICON_SCHEMA = pa.schema([
    pa.field("term_id",          pa.string()),
    pa.field("term",             pa.string()),
    pa.field("topic",            pa.string()),
    pa.field("observations",     pa.int32()),
    pa.field("confidence_sum",   pa.float32()),
    pa.field("avg_confidence",   pa.float32()),
    pa.field("status",           pa.string()),   # candidate | active | suppressed
    pa.field("consecutive_low",  pa.int32()),
    pa.field("last_updated",     pa.string()),   # ISO8601 UTC
])


# ── Term extraction ───────────────────────────────────────────────────────────


def _extract_terms(query: str) -> list[str]:
    """
    Extract unigrams and bigrams from query.
    Filters stop words and short tokens.
    """
    tokens = [
        t.lower().strip(".,!?;:'\"()")
        for t in query.split()
        if t.isalpha() and len(t) >= 4
    ]
    tokens = [t for t in tokens if t not in STOP_WORDS]

    unigrams = tokens
    bigrams  = [
        f"{tokens[i]} {tokens[i + 1]}"
        for i in range(len(tokens) - 1)
    ]

    # dedupe, preserve order
    seen = set()
    result = []
    for t in (unigrams + bigrams):
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def _term_id(term: str, topic: str) -> str:
    return hashlib.md5(f"{term}:{topic}".encode()).hexdigest()[:16]


# ── Weight computation ────────────────────────────────────────────────────────


def _compute_weight(avg_confidence: float, observations: int) -> float:
    """
    term_weight = avg_confidence * log10(observations + 1) * obs_factor
    obs_factor  = min(1.0, observations / MIN_OBS_FULL_WEIGHT)

    Dampened for low‑observation terms.
    """
    obs_factor   = min(1.0, observations / MIN_OBS_FULL_WEIGHT)
    term_weight  = avg_confidence * log(observations + 1, 10) * obs_factor
    return round(term_weight, 4)


# ── Decay ─────────────────────────────────────────────────────────────────────


def _apply_decay(
    avg_confidence: float,
    confidence_sum: float,
    last_updated: str,
    now: datetime,
) -> tuple[float, float]:
    """
    Apply time‑based decay proportional to days elapsed since last update.
    Floors avg_confidence at AVG_CONFIDENCE_FLOOR.
    """
    try:
        last = datetime.fromisoformat(last_updated)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        days = (now - last).total_seconds() / 86400.0
    except Exception:
        days = 0.0

    if days <= 0:
        return avg_confidence, confidence_sum

    factor            = DECAY_FACTOR ** days
    avg_confidence    = max(AVG_CONFIDENCE_FLOOR, avg_confidence * factor)
    confidence_sum    = max(0.0, confidence_sum * factor)

    return round(avg_confidence, 4), round(confidence_sum, 4)


# ── Transition logic ──────────────────────────────────────────────────────────


def _next_status(
    current_status: str,
    observations: int,
    avg_confidence: float,
    consecutive_low: int,
    last_updated: str,
) -> str:
    """
    State machine for term status transitions.

    candidate → active       : observations >= GRADUATION_OBSERVATIONS
                             AND avg_confidence >= GRADUATION_THRESHOLD
    active    → candidate    : 5 consecutive low scores
    candidate → suppressed   : 10 consecutive low scores
    active    → suppressed   : 13 consecutive low scores
    suppressed → candidate   : handled separately (probation/exploration)
    """
    if current_status == "candidate":
        if observations >= GRADUATION_OBSERVATIONS and avg_confidence >= GRADUATION_THRESHOLD:
            return "active"
        if consecutive_low >= SUPPRESSION_FROM_CAND:
            return "suppressed"

    elif current_status == "active":
        if consecutive_low >= SUPPRESSION_FROM_ACTIVE:
            return "suppressed"
        if consecutive_low >= DEMOTION_CONSECUTIVE:
            return "candidate"

    return current_status


def _is_probation_eligible(last_updated: str, now: datetime) -> bool:
    """Check if a suppressed term has served its probation period."""
    try:
        last = datetime.fromisoformat(last_updated)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        days = (now - last).total_seconds() / 86400.0
        return days >= SUPPRESSED_PROBATION_DAYS
    except Exception:
        return False


# ── LexiconStore ──────────────────────────────────────────────────────────────


class LexiconStore:
    """
    Learned keyword lexicon backed by LanceDB.

    Observes query terms correlated with high/low confidence scrapes
    and gradually promotes high‑signal terms into the active lexicon
    for each topic. Used exclusively for query expansion — never for
    content filtering or chunk ranking.

    State machine:
        candidate → active       (8 obs, avg_conf >= 0.70)
        active    → candidate    (5 consecutive lows)
        candidate → suppressed   (10 consecutive lows)
        active    → suppressed   (13 consecutive lows)
        suppressed → candidate   (30‑day probation OR rare exploration)
    """

    def __init__(self, db: lancedb.DBConnection):
        self.db     = db
        self._table = self._init_table()

    def _init_table(self):
        if LEXICON_TABLE not in self.db.table_names():
            return self.db.create_table(
                LEXICON_TABLE,
                schema=LEXICON_SCHEMA,
                mode="overwrite",
            )
        return self.db.open_table(LEXICON_TABLE)

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, term: str, topic: str) -> Optional[Dict[str, Any]]:
        """Fetch a single term record by term+topic."""
        try:
            tid     = _term_id(term, topic)
            results = (
                self._table.search()
                .where(f"term_id = '{tid}'")
                .limit(1)
                .to_list()
            )
            return results[0] if results else None
        except Exception:
            logger.exception("LexiconStore.get failed for term=%s topic=%s", term, topic)
            return None

    def get_active_terms(self, topic: str) -> List[Dict[str, Any]]:
        """
        Return all active terms for a topic, sorted by weight descending.
        """
        try:
            results = (
                self._table.search()
                .where(f"topic = '{topic}' AND status = 'active'")
                .to_list()
            )
            now = datetime.now(timezone.utc)
            weighted = []
            for r in results:
                avg_conf, conf_sum = _apply_decay(
                    float(r["avg_confidence"]),
                    float(r["confidence_sum"]),
                    r["last_updated"],
                    now,
                )
                weight = _compute_weight(avg_conf, int(r["observations"]))
                weighted.append({**r, "avg_confidence": avg_conf, "weight": weight})

            return sorted(weighted, key=lambda x: x["weight"], reverse=True)
        except Exception:
            logger.exception("LexiconStore.get_active_terms failed for topic=%s", topic)
            return []

    # ── Observe ───────────────────────────────────────────────────────────────

    def observe(
        self,
        query: str,
        topic: str,
        confidence: float,
    ) -> None:
        """
        Observe a query+confidence pair and update all extracted terms.
        Called after each scrape by ReputationStore.

        Args:
            query:     original search query
            topic:     inferred topic
            confidence: retrieval_confidence from ReputationStore.update()
        """
        terms = _extract_terms(query)
        if not terms:
            return

        is_high = confidence >= GRADUATION_THRESHOLD
        is_low  = confidence < LOW_CONFIDENCE_THRESHOLD
        now     = datetime.now(timezone.utc)

        for term in terms:
            try:
                self._observe_term(term, topic, confidence, is_high, is_low, now)
            except Exception:
                logger.exception(
                    "LexiconStore.observe failed for term=%s topic=%s", term, topic
                )

    def _observe_term(
        self,
        term: str,
        topic: str,
        confidence: float,
        is_high: bool,
        is_low: bool,
        now: datetime,
    ) -> None:
        existing = self.get(term, topic)
        tid      = _term_id(term, topic)

        if existing:
            # Apply decay first
            avg_conf, conf_sum = _apply_decay(
                float(existing["avg_confidence"]),
                float(existing["confidence_sum"]),
                existing["last_updated"],
                now,
            )

            observations   = int(existing["observations"]) + 1
            conf_sum       = conf_sum + confidence
            avg_conf       = conf_sum / observations
            avg_conf       = max(AVG_CONFIDENCE_FLOOR, avg_conf)
            current_status = existing["status"]
            consec_low     = int(existing["consecutive_low"])

            # Update consecutive_low streak
            if is_low:
                consec_low += 1
            elif is_high:
                consec_low = 0  # reset on high score

            # Handle suppressed recovery
            if current_status == "suppressed":
                if _is_probation_eligible(existing["last_updated"], now):
                    current_status = "candidate"
                    consec_low     = 0
                    logger.info(
                        "Term '%s/%s' recovered from suppression via probation",
                        term, topic,
                    )
                else:
                    # Always update stats in suppression; only explore with probability
                    if random.random() < EXPLORATION_PROBABILITY:
                        current_status = "candidate"
                        consec_low     = max(0, consec_low - 2)
                        logger.info(
                            "Term '%s/%s' recovered from suppression via exploration",
                            term, topic,
                        )

                    # Regardless of recovery, write updated stats with current_status
                    new_status = current_status
                    self._upsert(tid, term, topic, observations, conf_sum,
                                 avg_conf, new_status, consec_low, now)
                    return

            new_status = _next_status(
                current_status, observations, avg_conf, consec_low,
                existing["last_updated"],
            )

            # Only log when actually transitioning
            if new_status != current_status:
                logger.info(
                    "Term '%s/%s' transitioned %s → %s (obs=%d, avg_conf=%.3f, consec_low=%d)",
                    term, topic, current_status, new_status,
                    observations, avg_conf, consec_low,
                )

        else:
            # New term
            observations   = 1
            conf_sum       = confidence
            avg_conf       = max(AVG_CONFIDENCE_FLOOR, confidence)
            current_status = "candidate"
            consec_low     = 1 if is_low else 0
            new_status     = "candidate"

        self._upsert(tid, term, topic, observations, conf_sum,
                     avg_conf, new_status, consec_low, now)

    def _upsert(
        self,
        tid: str,
        term: str,
        topic: str,
        observations: int,
        confidence_sum: float,
        avg_confidence: float,
        status: str,
        consecutive_low: int,
        now: datetime,
    ) -> None:
        """
        Delete existing record (if any) and insert updated one.
        Can later be replaced with `merge_insert` if you want to avoid DELETE + ADD.
        """
        try:
            self._table.delete(f"term_id = '{tid}'")
        except Exception:
            pass  # term didn’t exist or table issue is non‑critical here

        self._table.add([{
            "term_id":          tid,
            "term":             term,
            "topic":            topic,
            "observations":     observations,
            "confidence_sum":   round(float(confidence_sum), 4),
            "avg_confidence":   round(float(avg_confidence), 4),
            "status":           status,
            "consecutive_low":  consecutive_low,
            "last_updated":     now.isoformat(),
        }])

    # ── Query expansion ───────────────────────────────────────────────────────

    def expand_query(
        self,
        query: str,
        topic: str,
        ambiguous: bool = False,
    ) -> str:
        """
        Expand query with up to MAX_LEXICON_TERMS active high‑weight terms.
        Only active terms are used. Suppressed and candidate terms are excluded.
        Lexicon terms are appended — original query always leads.

        Args:
            query:      original search query
            topic:      inferred topic
            ambiguous:  if True, also consider top terms from adjacent topic

        Returns:
            Expanded query string.
        """
        active = self.get_active_terms(topic)

        # Light cross‑topic for ambiguous queries — top 1 term from adjacent topic
        if ambiguous:
            adjacent = self._get_adjacent_topic(topic)
            if adjacent:
                adj_terms = self.get_active_terms(adjacent)
                if adj_terms:
                    active = active + [adj_terms[0]]

        # Filter by minimum weight and exclude terms already in query
        query_lower = query.lower()
        candidates  = [
            t for t in active
            if t.get("weight", 0.0) >= MIN_TERM_WEIGHT
            and t["term"].lower() not in query_lower
        ]

        # Top‑k by weight
        selected = candidates[:MAX_LEXICON_TERMS]
        if not selected:
            return query

        expansion   = " ".join(t["term"] for t in selected)
        final_query = f"{query} {expansion}"

        logger.debug(
            "Query expanded: '%s' → '%s' (topic=%s, terms=%d)",
            query, final_query, topic, len(selected),
        )

        return final_query

    def _get_adjacent_topic(self, topic: str) -> Optional[str]:
        """
        Returns a semantically adjacent topic for light cross‑topic expansion.
        Only for genuinely ambiguous topic pairs.
        """
        adjacency = {
            "finance":  "news",
            "news":     "finance",
            "science":  "tech",
            "tech":     "science",
            "health":   "science",
            "sports":   "news",
            "education": "tech",
        }
        return adjacency.get(topic)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def summary(self, topic: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return all terms sorted by weight, optionally filtered by topic.
        """
        try:
            all_records = self._table.to_pandas().to_dict(orient="records")
            if topic:
                all_records = [r for r in all_records if r.get("topic") == topic]

            now = datetime.now(timezone.utc)
            result = []
            for r in all_records:
                avg_conf, conf_sum = _apply_decay(
                    float(r["avg_confidence"]),
                    float(r["confidence_sum"]),
                    r["last_updated"],
                    now,
                )
                weight = _compute_weight(avg_conf, int(r["observations"]))
                result.append({
                    "term":            r["term"],
                    "topic":           r["topic"],
                    "weight":          weight,
                    "observations":    r["observations"],
                    "avg_confidence":  avg_conf,
                    "confidence_sum":  conf_sum,
                    "status":          r["status"],
                    "consecutive_low": r["consecutive_low"],
                })

            return sorted(result, key=lambda r: r["weight"], reverse=True)

        except Exception as e:
            logger.error("Lexicon summary failed: %s", e)
            return []