import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_CONTEXT_TOKENS      = 3000
MAX_CHUNKS              = 10
MIN_TOKENS_PER_CHUNK    = 80
MAX_TOKENS_PER_CHUNK    = 400
MAX_CHUNKS_PER_DOMAIN   = 3
MIN_CHUNK_SCORE         = 0.25    # below this = dropped before allocation
TOKEN_ESTIMATE_RATIO    = 4       # chars per token estimate

# Boilerplate patterns to strip during trimming
TRIM_PATTERNS = [
    r"(share this|follow us|subscribe|click here).*",
    r"(advertisement|sponsored content).*",
    r"(all rights reserved|copyright ©).*",
    r"(related articles?|see also|read more).*",
    r"\s{3,}",                    # excessive whitespace
]
TRIM_COMPILED = [re.compile(p, re.IGNORECASE) for p in TRIM_PATTERNS]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // TOKEN_ESTIMATE_RATIO)


def _get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "unknown"


def _trim_to_tokens(text: str, max_tokens: int) -> str:
    """
    Trim text to approximately max_tokens.
    Trims at sentence boundary where possible.
    """
    max_chars = max_tokens * TOKEN_ESTIMATE_RATIO
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]

    # Try to trim at last sentence boundary
    last_boundary = max(
        truncated.rfind(". "),
        truncated.rfind("! "),
        truncated.rfind("? "),
        truncated.rfind("\n"),
    )

    if last_boundary > max_chars * 0.6:
        return truncated[:last_boundary + 1].strip()

    return truncated.strip()


def _strip_boilerplate(text: str) -> str:
    """Remove trailing boilerplate patterns from chunk text."""
    for pattern in TRIM_COMPILED:
        text = pattern.sub("", text)
    return text.strip()


def _near_duplicate(a: str, b: str, threshold: float = 0.80) -> bool:
    """
    Shingling-based near-duplicate check.
    Uses character 4-grams for fast comparison.
    """
    def shingles(text: str, k: int = 4) -> set:
        t = text.lower().strip()
        return set(t[i:i + k] for i in range(len(t) - k + 1))

    s1, s2 = shingles(a), shingles(b)
    if not s1 or not s2:
        return False
    return len(s1 & s2) / len(s1 | s2) >= threshold


# ── Output dataclasses ────────────────────────────────────────────────────────

@dataclass
class ContextChunk:
    """
    A single chunk selected for inclusion in the context window.
    Carries scoring, source attribution, and allocated token budget.
    """
    content:          str
    source:           str
    domain:           str
    title:            str
    score:            float        # composite quality score
    allocated_tokens: int
    chunk_index:      int
    coherence:        float
    previously_used:  bool = False

    @property
    def token_estimate(self) -> int:
        return _estimate_tokens(self.content)


@dataclass
class BuiltContext:
    """
    Output of ContextBuilder.build().
    Contains structured chunks and metadata.
    Exposes to_prompt() for LLM-ready string output.
    """
    chunks:        list[ContextChunk]
    total_tokens:  int
    sources:       list[str]
    query:         str
    topic:         str
    token_budget:  int
    truncated:     bool = False     # True if chunks were dropped due to overflow

    def to_prompt(self) -> str:
        """
        Render context as a structured LLM-ready string.

        Format per chunk:
            [Source: domain.com | Score: 0.82 | Chunk: 2]
            <content>
        """
        if not self.chunks:
            return ""

        parts = []
        for chunk in self.chunks:
            header = (
                f"[Source: {chunk.domain} | "
                f"Score: {chunk.score:.2f} | "
                f"Chunk: {chunk.chunk_index}]"
            )
            parts.append(f"{header}\n{chunk.content}")

        return "\n\n".join(parts)

    def to_dict(self) -> dict:
        """Structured dict output for programmatic use."""
        return {
            "chunks":       [
                {
                    "content":    c.content,
                    "source":     c.source,
                    "domain":     c.domain,
                    "title":      c.title,
                    "score":      c.score,
                    "tokens":     c.allocated_tokens,
                    "chunk_index": c.chunk_index,
                }
                for c in self.chunks
            ],
            "total_tokens":  self.total_tokens,
            "sources":       self.sources,
            "query":         self.query,
            "topic":         self.topic,
            "token_budget":  self.token_budget,
            "truncated":     self.truncated,
        }

    def summary(self) -> dict:
        return {
            "chunks":        len(self.chunks),
            "total_tokens":  self.total_tokens,
            "sources":       len(self.sources),
            "truncated":     self.truncated,
            "utilization":   round(self.total_tokens / self.token_budget, 3)
                             if self.token_budget else 0.0,
        }


# ── ContextBuilder ────────────────────────────────────────────────────────────

class ContextBuilder:
    """
    Relevance-aware, token-budgeted, session-aware context composer.

    Pipeline:
    1. Score chunks (composite quality + reputation signal)
    2. Deduplicate near-identical chunks
    3. Filter below minimum score threshold
    4. Enforce domain diversity (max N chunks per domain)
    5. Allocate token budget proportional to score
    6. Trim each chunk to allocated budget at sentence boundary
    7. Strip boilerplate from trimmed content
    8. Enforce total token cap — drop lowest-score chunks if overflow
    9. Structure output with source attribution

    Session awareness:
    - Tracks used chunk IDs across calls within a session
    - Penalizes previously used chunks (still included if highly relevant)
    - Reuses high-value chunks if score exceeds reuse threshold
    """

    def __init__(
        self,
        max_context_tokens:   int   = MAX_CONTEXT_TOKENS,
        max_chunks:           int   = MAX_CHUNKS,
        min_tokens_per_chunk: int   = MIN_TOKENS_PER_CHUNK,
        max_tokens_per_chunk: int   = MAX_TOKENS_PER_CHUNK,
        max_chunks_per_domain: int  = MAX_CHUNKS_PER_DOMAIN,
        min_chunk_score:      float = MIN_CHUNK_SCORE,
        reuse_threshold:      float = 0.75,   # reuse previously used chunk if score >= this
        reuse_penalty:        float = 0.15,   # score penalty for previously used chunks
    ):
        self.max_context_tokens    = max_context_tokens
        self.max_chunks            = max_chunks
        self.min_tokens_per_chunk  = min_tokens_per_chunk
        self.max_tokens_per_chunk  = max_tokens_per_chunk
        self.max_chunks_per_domain = max_chunks_per_domain
        self.min_chunk_score       = min_chunk_score
        self.reuse_threshold       = reuse_threshold
        self.reuse_penalty         = reuse_penalty

        # Session state — tracks used chunk fingerprints
        self._used_chunks: set[str] = set()

    def reset_session(self) -> None:
        """Clear session memory. Call between unrelated queries."""
        self._used_chunks.clear()

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(
        self,
        chunks:        list[dict],
        query:         str,
        topic:         str,
        token_budget:  Optional[int]        = None,
        reputation:    Optional[dict]       = None,
        session_chunks: Optional[list[dict]] = None,
    ) -> BuiltContext:
        """
        Build the optimal context from ranked chunks within token budget.

        Args:
            chunks:         ranked chunks from reranker (list of dicts)
            query:          original search query
            topic:          inferred topic
            token_budget:   override max_context_tokens if provided
            reputation:     dict of {domain: confidence} for score boosting
            session_chunks: previously used chunks from other sessions
                            (for cross-session awareness)

        Returns:
            BuiltContext with to_prompt() and to_dict() methods.
        """
        budget = min(
            token_budget or self.max_context_tokens,
            self.max_context_tokens,
        )

        if not chunks:
            return self._empty(query, topic, budget)

        # ── Step 1: Score ─────────────────────────────────────────────────────
        scored = self._score_chunks(chunks, reputation or {})

        # ── Step 2: Deduplicate ───────────────────────────────────────────────
        scored = self._deduplicate(scored)

        # ── Step 3: Filter below min score ───────────────────────────────────
        scored = [c for c in scored if c["_score"] >= self.min_chunk_score]

        if not scored:
            return self._empty(query, topic, budget)

        # ── Step 4: Domain diversity ──────────────────────────────────────────
        scored = self._apply_diversity(scored)

        # ── Step 5: Cap at max_chunks ─────────────────────────────────────────
        scored = scored[:self.max_chunks]

        # ── Step 6: Token allocation ──────────────────────────────────────────
        allocated = self._allocate_tokens(scored, budget)

        # ── Step 7 + 8: Trim + strip boilerplate ─────────────────────────────
        context_chunks = []
        total_tokens   = 0

        for item in allocated:
            content = item["content"]
            content = _strip_boilerplate(content)
            content = _trim_to_tokens(content, item["_allocated_tokens"])

            if _estimate_tokens(content) < self.min_tokens_per_chunk:
                continue

            domain   = _get_domain(item.get("source", ""))
            fp       = self._fingerprint(content)
            prev_used = fp in self._used_chunks

            ctx_chunk = ContextChunk(
                content          = content,
                source           = item.get("source", ""),
                domain           = domain,
                title            = item.get("title", ""),
                score            = round(item["_score"], 4),
                allocated_tokens = item["_allocated_tokens"],
                chunk_index      = int(item.get("chunk_index", 0)),
                coherence        = float(item.get("coherence_score", 0.0)),
                previously_used  = prev_used,
            )

            total_tokens += ctx_chunk.token_estimate
            context_chunks.append(ctx_chunk)
            self._used_chunks.add(fp)

        # ── Step 9: Enforce total token cap ───────────────────────────────────
        context_chunks, truncated = self._enforce_cap(context_chunks, budget)
        total_tokens = sum(_estimate_tokens(c.content) for c in context_chunks)

        sources = list(dict.fromkeys(
            c.source for c in context_chunks if c.source
        ))

        logger.info(
            "ContextBuilder: query=%r chunks=%d tokens=%d/%d sources=%d truncated=%s",
            query, len(context_chunks), total_tokens, budget,
            len(sources), truncated,
        )

        return BuiltContext(
            chunks       = context_chunks,
            total_tokens = total_tokens,
            sources      = sources,
            query        = query,
            topic        = topic,
            token_budget = budget,
            truncated    = truncated,
        )

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_chunks(
        self,
        chunks:     list[dict],
        reputation: dict[str, float],
    ) -> list[dict]:
        """
        Compute composite score per chunk.
        Score = 0.5 * reranker_score + 0.3 * coherence + 0.2 * reputation_boost
        Apply session penalty for previously used chunks.
        """
        scored = []
        total  = len(chunks)

        for i, chunk in enumerate(chunks):
            coherence = float(chunk.get("coherence_score", 0.0))
            domain    = _get_domain(chunk.get("source", ""))

            # Reranker position score — higher rank = higher score
            rank_score = 1.0 - (i / max(total, 1))

            # Reputation boost
            rep_score = reputation.get(domain, 0.5)

            # Composite
            score = (
                0.50 * rank_score  +
                0.30 * coherence   +
                0.20 * rep_score
            )

            # Session penalty for previously used chunks
            fp = self._fingerprint(chunk.get("content", ""))
            if fp in self._used_chunks:
                score -= self.reuse_penalty
                if score < self.reuse_threshold:
                    # Not worth reusing — skip
                    continue

            scored.append({**chunk, "_score": round(score, 4)})

        return sorted(scored, key=lambda x: x["_score"], reverse=True)

    # ── Deduplication ─────────────────────────────────────────────────────────

    def _deduplicate(self, chunks: list[dict]) -> list[dict]:
        """Remove near-duplicate chunks using shingling."""
        deduped  = []
        contents = []

        for chunk in chunks:
            content = chunk.get("content", "")
            if not any(_near_duplicate(content, existing) for existing in contents[-10:]):
                deduped.append(chunk)
                contents.append(content)

        return deduped

    # ── Domain diversity ──────────────────────────────────────────────────────

    def _apply_diversity(self, chunks: list[dict]) -> list[dict]:
        """Enforce max_chunks_per_domain."""
        domain_counts: dict[str, int] = {}
        result = []

        for chunk in chunks:
            domain = _get_domain(chunk.get("source", ""))
            count  = domain_counts.get(domain, 0)
            if count < self.max_chunks_per_domain:
                result.append(chunk)
                domain_counts[domain] = count + 1

        return result

    # ── Token allocation ──────────────────────────────────────────────────────

    def _allocate_tokens(
        self,
        chunks: list[dict],
        budget: int,
    ) -> list[dict]:
        """
        Allocate token budget proportionally by score.
        Each chunk gets: clamp(weight * budget, min_tokens, max_tokens)
        """
        total_score = sum(c["_score"] for c in chunks)
        if total_score == 0:
            total_score = 1.0

        result = []
        for chunk in chunks:
            weight    = chunk["_score"] / total_score
            allocated = int(weight * budget)
            allocated = max(self.min_tokens_per_chunk,
                          min(self.max_tokens_per_chunk, allocated))
            result.append({**chunk, "_allocated_tokens": allocated})

        return result

    # ── Token cap enforcement ─────────────────────────────────────────────────

    def _enforce_cap(
        self,
        chunks:  list[ContextChunk],
        budget:  int,
    ) -> tuple[list[ContextChunk], bool]:
        """
        Drop lowest-score chunks until total tokens fit within budget.
        Returns (kept_chunks, truncated_flag).
        """
        total     = sum(_estimate_tokens(c.content) for c in chunks)  # ← recompute from actual content
        truncated = False

        if total <= budget:
            return chunks, False

        # Sort by score descending, drop from bottom
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
        kept          = []
        running_total = 0

        for chunk in sorted_chunks:
            tokens = _estimate_tokens(chunk.content)
            if running_total + tokens <= budget:
                kept.append(chunk)
                running_total += tokens
            else:
                truncated = True

        # Restore original order
        original_order = {id(c): i for i, c in enumerate(chunks)}
        kept.sort(key=lambda c: original_order.get(id(c), 0))

        return kept, truncated

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fingerprint(self, content: str) -> str:
        """Short fingerprint for session dedup."""
        import hashlib
        return hashlib.md5(content[:200].encode()).hexdigest()[:12]

    def _empty(self, query: str, topic: str, budget: int) -> BuiltContext:
        return BuiltContext(
            chunks       = [],
            total_tokens = 0,
            sources      = [],
            query        = query,
            topic        = topic,
            token_budget = budget,
            truncated    = False,
        )