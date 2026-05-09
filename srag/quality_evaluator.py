import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_CONTENT_CHARS       = 150    # below this = not useful
MIN_WORD_COUNT          = 20     # below this = not useful
MAX_SHORT_WORD_RATIO    = 0.60   # too many short words = nav/boilerplate
MIN_COHERENCE           = 0.30   # below this = low quality chunk
HIGH_COHERENCE          = 0.65   # above this = high quality chunk
MIN_USEFUL_SENTENCES    = 2      # chunk must have at least this many sentences

# Noise signal patterns — presence degrades usefulness score
NOISE_SIGNALS = [
    r"cookie",
    r"javascript",
    r"enable javascript",
    r"switch language",
    r"all rights reserved",
    r"subscribe now",
    r"sign up",
    r"log in",
    r"advertisement",
    r"sponsored",
    r"click here",
    r"privacy policy",
    r"terms of service",
    r"404",
    r"page not found",
]
NOISE_COMPILED = [re.compile(p, re.IGNORECASE) for p in NOISE_SIGNALS]


# ── Output dataclasses ────────────────────────────────────────────────────────

@dataclass
class ChunkQuality:
    """
    Quality evaluation result for a single chunk.
    """
    chunk_index:     int
    source:          str
    coherence:       float
    usefulness:      float        # 0.0 - 1.0 composite score
    noise_detected:  bool
    word_count:      int
    sentence_count:  int
    passed:          bool         # True if chunk meets minimum bar

    @property
    def composite_score(self) -> float:
        """
        Weighted composite of coherence and usefulness.
        Coherence is structural quality, usefulness is content quality.
        """
        return round(0.6 * self.coherence + 0.4 * self.usefulness, 4)


@dataclass
class DocumentQuality:
    """
    Aggregated quality evaluation for all chunks from one document/domain.
    """
    domain:           str
    total_chunks:     int
    passed_chunks:    int
    avg_coherence:    float
    avg_usefulness:   float
    avg_composite:    float
    noise_rate:       float       # fraction of chunks with noise detected
    chunk_results:    list[ChunkQuality] = field(default_factory=list)

    @property
    def useful_hit_rate(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return round(self.passed_chunks / self.total_chunks, 4)

    @property
    def irrelevance_rate(self) -> float:
        return round(1.0 - self.useful_hit_rate, 4)

    @property
    def is_useful_source(self) -> bool:
        """True if domain produced enough useful chunks to be worth indexing."""
        return self.passed_chunks >= 1 and self.avg_composite >= MIN_COHERENCE


@dataclass
class SessionQuality:
    """
    Aggregated quality across all documents in a search session.
    Keyed by domain for reputation update.
    """
    query:            str
    topic:            str
    doc_results:      dict[str, DocumentQuality] = field(default_factory=dict)

    def get_domain_signals(self, domain: str) -> Optional[dict]:
        """
        Return reputation-ready signals for a domain.
        Returns None if domain not present.
        """
        doc = self.doc_results.get(domain)
        if doc is None:
            return None
        return {
            "avg_chunk_quality": doc.avg_coherence,
            "useful_hit_rate":   doc.useful_hit_rate,
            "irrelevance_rate":  doc.irrelevance_rate,
        }

    def summary(self) -> dict:
        total_chunks  = sum(d.total_chunks  for d in self.doc_results.values())
        passed_chunks = sum(d.passed_chunks for d in self.doc_results.values())
        useful_domains = sum(
            1 for d in self.doc_results.values() if d.is_useful_source
        )
        return {
            "query":          self.query,
            "topic":          self.topic,
            "domains":        len(self.doc_results),
            "useful_domains": useful_domains,
            "total_chunks":   total_chunks,
            "passed_chunks":  passed_chunks,
            "pass_rate":      round(passed_chunks / total_chunks, 4) if total_chunks else 0.0,
        }


# ── QualityEvaluator ──────────────────────────────────────────────────────────

class QualityEvaluator:
    """
    Evaluates chunk and document quality for the SRag pipeline.

    Responsibilities:
    - Score individual chunks on coherence + usefulness
    - Detect noise signals (boilerplate, cookie walls, nav fragments)
    - Aggregate per-domain quality stats for reputation updates
    - Produce SessionQuality for orchestrator to pass to ReputationStore

    NOT responsible for:
    - Filtering chunks (indexer does that via min_coherence)
    - Reranking (CrossEncoder reranker handles that)
    - Content deduplication (scraper and chunker handle that)
    """

    def __init__(
        self,
        min_coherence:       float = MIN_COHERENCE,
        min_content_chars:   int   = MIN_CONTENT_CHARS,
        min_word_count:      int   = MIN_WORD_COUNT,
        min_useful_sentences: int  = MIN_USEFUL_SENTENCES,
    ):
        self.min_coherence        = min_coherence
        self.min_content_chars    = min_content_chars
        self.min_word_count       = min_word_count
        self.min_useful_sentences = min_useful_sentences

    # ── Chunk evaluation ──────────────────────────────────────────────────────

    def evaluate_chunk(self, chunk: dict) -> ChunkQuality:
        """
        Evaluate a single chunk dict from SmartChunker output.

        Args:
            chunk: dict with keys: content, coherence_score,
                   chunk_index, source, sentence_count

        Returns:
            ChunkQuality with composite score and pass/fail.
        """
        content        = chunk.get("content", "")
        coherence      = float(chunk.get("coherence_score", 0.0))
        chunk_index    = int(chunk.get("chunk_index", 0))
        source         = chunk.get("source", "")
        sentence_count = int(chunk.get("sentence_count", 0))

        usefulness, noise_detected = self._score_usefulness(content)

        passed = (
            coherence      >= self.min_coherence
            and usefulness >= 0.3
            and len(content) >= self.min_content_chars
            and not noise_detected
        )

        return ChunkQuality(
            chunk_index    = chunk_index,
            source         = source,
            coherence      = coherence,
            usefulness     = usefulness,
            noise_detected = noise_detected,
            word_count     = len(content.split()),
            sentence_count = sentence_count,
            passed         = passed,
        )

    # ── Document evaluation ───────────────────────────────────────────────────

    def evaluate_document(
        self,
        chunks: list[dict],
        domain: str,
    ) -> DocumentQuality:
        """
        Evaluate all chunks from a single domain/document.

        Args:
            chunks: list of chunk dicts from SmartChunker for this domain
            domain: domain string e.g. "realpython.com"

        Returns:
            DocumentQuality with aggregated stats.
        """
        if not chunks:
            return DocumentQuality(
                domain         = domain,
                total_chunks   = 0,
                passed_chunks  = 0,
                avg_coherence  = 0.0,
                avg_usefulness = 0.0,
                avg_composite  = 0.0,
                noise_rate     = 0.0,
            )

        chunk_results = [self.evaluate_chunk(c) for c in chunks]
        total         = len(chunk_results)
        passed        = sum(1 for r in chunk_results if r.passed)
        noise_count   = sum(1 for r in chunk_results if r.noise_detected)

        avg_coherence  = sum(r.coherence        for r in chunk_results) / total
        avg_usefulness = sum(r.usefulness       for r in chunk_results) / total
        avg_composite  = sum(r.composite_score  for r in chunk_results) / total

        logger.debug(
            "QualityEvaluator: %s | %d chunks, %d passed, "
            "avg_coherence=%.3f avg_usefulness=%.3f",
            domain, total, passed, avg_coherence, avg_usefulness,
        )

        return DocumentQuality(
            domain         = domain,
            total_chunks   = total,
            passed_chunks  = passed,
            avg_coherence  = round(avg_coherence,  4),
            avg_usefulness = round(avg_usefulness, 4),
            avg_composite  = round(avg_composite,  4),
            noise_rate     = round(noise_count / total, 4),
            chunk_results  = chunk_results,
        )

    # ── Session evaluation ────────────────────────────────────────────────────

    def evaluate_session(
        self,
        chunks:  list[dict],
        query:   str,
        topic:   str,
    ) -> SessionQuality:
        """
        Evaluate all chunks across a full search session.
        Groups chunks by domain and produces per-domain DocumentQuality.

        Args:
            chunks: all chunks from chunker.chunk_docs() for this session
            query:  original search query
            topic:  inferred topic from TopicClassifier

        Returns:
            SessionQuality with per-domain breakdown.
        """
        # Group chunks by domain
        from urllib.parse import urlparse

        def get_domain(url: str) -> str:
            try:
                return urlparse(url).netloc.replace("www.", "")
            except Exception:
                return "unknown"

        by_domain: dict[str, list[dict]] = {}
        for chunk in chunks:
            domain = get_domain(chunk.get("source", ""))
            by_domain.setdefault(domain, []).append(chunk)

        session = SessionQuality(query=query, topic=topic)
        for domain, domain_chunks in by_domain.items():
            session.doc_results[domain] = self.evaluate_document(
                domain_chunks, domain
            )

        logger.info(
            "QualityEvaluator session: %s",
            session.summary(),
        )

        return session

    # ── Usefulness scoring ────────────────────────────────────────────────────

    def _score_usefulness(self, text: str) -> tuple[float, bool]:
        """
        Score content usefulness on [0, 1].
        Returns (score, noise_detected).

        Signals:
        - Length penalty for short content
        - Short word ratio penalty (nav fragments)
        - Noise pattern detection
        - Sentence structure bonus
        """
        if not text or not text.strip():
            return 0.0, False

        text_lower = text.lower().strip()

        # Noise detection
        noise_detected = any(p.search(text_lower) for p in NOISE_COMPILED)
        if noise_detected:
            return 0.1, True

        words       = text.split()
        word_count  = len(words)
        char_count  = len(text.strip())

        # Length score — sigmoid-like ramp
        if char_count < self.min_content_chars:
            length_score = char_count / self.min_content_chars * 0.5
        else:
            length_score = min(1.0, 0.5 + (char_count - self.min_content_chars) / 1000)

        # Short word penalty
        if word_count > 0:
            short_words       = sum(1 for w in words if len(w) <= 3)
            short_word_ratio  = short_words / word_count
            word_quality      = max(0.0, 1.0 - (short_word_ratio / MAX_SHORT_WORD_RATIO))
        else:
            word_quality = 0.0

        # Sentence structure bonus — more sentences = more content
        sentences       = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
        sentence_count  = len(sentences)
        sentence_bonus  = min(0.2, sentence_count * 0.02)

        # Composite usefulness
        usefulness = (
            0.45 * length_score  +
            0.35 * word_quality  +
            0.20 * min(1.0, sentence_bonus / 0.2)
        )

        return round(min(1.0, max(0.0, usefulness)), 4), False