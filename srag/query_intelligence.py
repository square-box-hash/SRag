import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_VARIANTS          = 5        # hard cap on total query variants
MIN_VARIANT_LENGTH    = 8        # discard variants shorter than this
MAX_VARIANT_LENGTH    = 180      # discard variants longer than this (DDGS URL limit)
CURRENT_YEAR          = "2026"
PREVIOUS_YEAR         = "2025"

# Topic-specific search constraints
TOPIC_SITE_HINTS: dict[str, list[str]] = {
    "tech":          ["site:docs.python.org", "site:stackoverflow.com", "site:github.com"],
    "finance":       ["site:rbi.org.in", "site:gst.gov.in", "site:incometax.gov.in"],
    "science":       ["site:arxiv.org", "site:ncbi.nlm.nih.gov", "site:nature.com"],
    "health":        ["site:ncbi.nlm.nih.gov", "site:who.int"],
    "education":     ["site:nptel.ac.in", "site:mit.edu"],
    "sports":        [],   # sports content is broad, no site hints
    "news":          [],   # news is broad, no site hints
    "entertainment": [],
    "general":       [],
}

# Synonym rephrase map
REPHRASE_MAP: dict[str, str] = {
    "tutorial":  "guide",
    "guide":     "tutorial",
    "how to":    "how do I",
    "error":     "issue fix",
    "fix":       "solution",
    "rate":      "percentage",
    "law":       "regulation",
    "price":     "cost",
    "best":      "top",
    "latest":    "recent",
    "vs":        "compared to",
    "explain":   "what is",
    "what is":   "explain",
    "overview":  "introduction",
}


# ── Output dataclasses ────────────────────────────────────────────────────────

@dataclass
class QueryVariant:
    """
    A single query variant with metadata about how it was generated.
    """
    text:     str
    strategy: str       # "original" | "year" | "rephrase" | "lexicon" | "site_hint"
    weight:   float     # higher = more likely to produce good results
    topic:    str

    def is_valid(self) -> bool:
        return (
            MIN_VARIANT_LENGTH <= len(self.text.strip()) <= MAX_VARIANT_LENGTH
            and bool(self.text.strip())
        )


@dataclass
class QueryPlan:
    """
    Full set of query variants for a search request.
    Bounded at MAX_VARIANTS.
    """
    original:  str
    topic:     str
    ambiguous: bool
    variants:  list[QueryVariant] = field(default_factory=list)

    def get_queries(self) -> list[str]:
        """Return deduplicated query strings, original first."""
        seen   = set()
        result = []
        for v in sorted(self.variants, key=lambda x: x.weight, reverse=True):
            text = v.text.strip()
            if text and text not in seen:
                seen.add(text)
                result.append(text)
        return result[:MAX_VARIANTS]

    def summary(self) -> dict:
        return {
            "original":  self.original,
            "topic":     self.topic,
            "ambiguous": self.ambiguous,
            "variants":  len(self.variants),
            "queries":   self.get_queries(),
        }


# ── QueryIntelligence ─────────────────────────────────────────────────────────

class QueryIntelligence:
    """
    Intelligent query rewriter and variant generator.

    Strategies (applied in order, bounded at MAX_VARIANTS=5):
    1. Original query (always included, weight=1.0)
    2. Year expansion (freshness signal)
    3. Synonym rephrase (broader coverage)
    4. Lexicon expansion (learned high-signal terms, topic-scoped)
    5. Site hint (topic-specific authoritative source)

    Hard constraints:
    - MAX_VARIANTS = 5 (DDGS + token budget)
    - MAX_VARIANT_LENGTH = 180 chars (DDGS URL limit)
    - Lexicon terms only from active status
    - Original query always leads

    Used exclusively in scraper.get_facts() — not for
    chunk filtering or reranking.
    """

    def __init__(self, lexicon=None):
        """
        Args:
            lexicon: LexiconStore instance or None.
                     If None, lexicon expansion is skipped gracefully.
        """
        self.lexicon = lexicon

    def rewrite(
        self,
        query:     str,
        topic:     str,
        ambiguous: bool = False,
        history:   Optional[list[str]] = None,
        constraints: Optional[dict]   = None,
    ) -> QueryPlan:
        """
        Generate a bounded set of query variants.

        Args:
            query:       original search query
            topic:       inferred topic from TopicClassifier
            ambiguous:   True if primary/secondary topic confidence gap < 0.15
            history:     previous queries in this session (avoid repetition)
            constraints: optional overrides e.g. {"max_variants": 3, "no_site_hints": True}

        Returns:
            QueryPlan with ranked, deduplicated variants.
        """
        if not query or not query.strip():
            logger.warning("QueryIntelligence.rewrite: empty query")
            return QueryPlan(original=query, topic=topic, ambiguous=ambiguous)

        constraints  = constraints or {}
        max_variants = min(
            MAX_VARIANTS,
            constraints.get("max_variants", MAX_VARIANTS)
        )
        history_set  = set(h.lower().strip() for h in (history or []))

        plan = QueryPlan(original=query, topic=topic, ambiguous=ambiguous)

        # ── Strategy 1: Original ──────────────────────────────────────────────
        plan.variants.append(QueryVariant(
            text     = query.strip(),
            strategy = "original",
            weight   = 1.0,
            topic    = topic,
        ))

        # ── Strategy 2: Year expansion ────────────────────────────────────────
        year_variant = self._year_expansion(query, topic)
        if year_variant:
            plan.variants.append(year_variant)

        # ── Strategy 3: Synonym rephrase ──────────────────────────────────────
        rephrase_variant = self._rephrase(query, topic)
        if rephrase_variant:
            plan.variants.append(rephrase_variant)

        # ── Strategy 4: Lexicon expansion ─────────────────────────────────────
        if self.lexicon and not constraints.get("no_lexicon"):
            lexicon_variant = self._lexicon_expand(query, topic, ambiguous)
            if lexicon_variant:
                plan.variants.append(lexicon_variant)

        # ── Strategy 5: Site hint ─────────────────────────────────────────────
        if not constraints.get("no_site_hints"):
            site_variant = self._site_hint(query, topic)
            if site_variant:
                plan.variants.append(site_variant)

        # Filter invalid + history duplicates
        plan.variants = [
            v for v in plan.variants
            if v.is_valid()
            and v.text.lower().strip() not in history_set
        ]

        # Deduplicate by text
        seen     = set()
        deduped  = []
        for v in plan.variants:
            key = v.text.lower().strip()
            if key not in seen:
                seen.add(key)
                deduped.append(v)
        plan.variants = deduped[:max_variants]

        logger.debug(
            "QueryIntelligence: '%s' → %d variants (topic=%s ambiguous=%s)",
            query, len(plan.variants), topic, ambiguous,
        )

        return plan

    # ── Strategies ────────────────────────────────────────────────────────────

    def _year_expansion(self, query: str, topic: str) -> Optional[QueryVariant]:
        """Add current year for freshness — skip if year already present."""
        if CURRENT_YEAR in query or PREVIOUS_YEAR in query:
            return None

        # Year expansion most useful for fast-changing topics
        high_value_topics = {"finance", "news", "sports", "tech", "health"}
        weight = 0.85 if topic in high_value_topics else 0.65

        return QueryVariant(
            text     = f"{query.strip()} {CURRENT_YEAR}",
            strategy = "year",
            weight   = weight,
            topic    = topic,
        )

    def _rephrase(self, query: str, topic: str) -> Optional[QueryVariant]:
        """
        Rephrase query using synonym map.
        Returns None if no substitution found.
        """
        q_lower    = query.lower()
        rephrased  = query

        for original, replacement in REPHRASE_MAP.items():
            if original in q_lower:
                rephrased = re.sub(
                    re.escape(original),
                    replacement,
                    query,
                    count=1,
                    flags=re.IGNORECASE,
                )
                break

        if rephrased.lower().strip() == query.lower().strip():
            return None

        return QueryVariant(
            text     = rephrased.strip(),
            strategy = "rephrase",
            weight   = 0.75,
            topic    = topic,
        )

    def _lexicon_expand(
        self,
        query:     str,
        topic:     str,
        ambiguous: bool,
    ) -> Optional[QueryVariant]:
        """
        Append top active lexicon terms to query.
        Uses LexiconStore.expand_query() — only active terms,
        max MAX_LEXICON_TERMS appended, original always leads.
        """
        try:
            expanded = self.lexicon.expand_query(
                query     = query,
                topic     = topic,
                ambiguous = ambiguous,
            )
            if expanded.lower().strip() == query.lower().strip():
                return None

            return QueryVariant(
                text     = expanded.strip(),
                strategy = "lexicon",
                weight   = 0.90,   # high weight — learned signal
                topic    = topic,
            )
        except Exception:
            logger.exception("QueryIntelligence._lexicon_expand failed")
            return None

    def _site_hint(self, query: str, topic: str) -> Optional[QueryVariant]:
        """
        Append a site: hint for authoritative sources by topic.
        Only for topics with defined site hints.
        Picks the first hint not already implied by the query.
        """
        hints = TOPIC_SITE_HINTS.get(topic, [])
        if not hints:
            return None

        q_lower = query.lower()
        for hint in hints:
            domain = hint.replace("site:", "")
            if domain not in q_lower:
                return QueryVariant(
                    text     = f"{query.strip()} {hint}",
                    strategy = "site_hint",
                    weight   = 0.70,
                    topic    = topic,
                )

        return None

    # ── Context injection (sequential search) ─────────────────────────────────

    def inject_context(
        self,
        query:   str,
        context: str,
        topic:   str,
    ) -> str:
        """
        Safely inject context from a previous session into a query.
        Extracts the first substantive line (skips author/date headers).
        Hard-capped at MAX_VARIANT_LENGTH.

        Used by sequential_search depends_on logic in orchestrator.
        Replaces the current ad-hoc context injection.
        """
        if not context or not context.strip():
            return query

        # Skip short header lines (author names, dates, nav fragments)
        lines = [
            l.strip() for l in context.splitlines()
            if len(l.strip()) > 40
        ]

        if not lines:
            return query

        core    = lines[0][:80]
        result  = f"{query} {core}"

        # Hard cap
        if len(result) > MAX_VARIANT_LENGTH:
            result = result[:MAX_VARIANT_LENGTH]

        logger.debug(
            "QueryIntelligence.inject_context: '%s' → '%s'",
            query, result,
        )

        return result