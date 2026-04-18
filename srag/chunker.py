import re
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


# ── Sentence splitter ─────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences respecting common boundaries.
    Handles abbreviations, decimals, and newline-based splits.
    """
    # Normalize excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # Split on sentence endings followed by space + capital
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Also split on double newlines (paragraph breaks)
    result = []
    for s in sentences:
        parts = s.split('\n\n')
        result.extend([p.strip() for p in parts if p.strip()])

    return result


def _estimate_tokens(text: str) -> int:
    """Rough token estimate — 1 token ≈ 4 chars."""
    return max(1, len(text) // 4)


# ── Noise detection ───────────────────────────────────────────────────────────

NOISE_PATTERNS = [
    r'^(switch language|cookie policy|accept cookies|privacy policy)',
    r'^(all rights reserved|copyright ©)',
    r'^(share this|follow us|subscribe)',
    r'^(advertisement|sponsored|related articles)',
    r'^\s*[\|\-—]{3,}\s*$',   # pure separator lines
    r'^\s*\d+\s*$',            # lone numbers (page numbers)
    r'^(home|about|contact|login|sign up|sign in)\s*$',
]

NOISE_COMPILED = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]


def _is_noise(sentence: str) -> bool:
    """Returns True if sentence is nav/boilerplate garbage."""
    if len(sentence.strip()) < 15:
        return True
    for pattern in NOISE_COMPILED:
        if pattern.match(sentence.strip()):
            return True
    return False


def _is_near_duplicate(a: str, b: str, threshold: float = 0.85) -> bool:
    """
    Shingling-based near-duplicate detection.
    Uses character 3-grams for fast comparison.
    """
    def shingles(text, k=3):
        text = text.lower().strip()
        return set(text[i:i+k] for i in range(len(text) - k + 1))

    s1, s2 = shingles(a), shingles(b)
    if not s1 or not s2:
        return False
    jaccard = len(s1 & s2) / len(s1 | s2)
    return jaccard >= threshold


# ── Semantic boundary detection ───────────────────────────────────────────────

def _find_semantic_boundaries(
    sentences: List[str],
    embeddings: np.ndarray,
    threshold: float = 0.75,
) -> List[int]:
    """
    Find indices where topic shifts occur using cosine similarity
    between adjacent sentence embeddings.
    Low similarity = topic boundary = good split point.
    """
    boundaries = []
    for i in range(1, len(embeddings)):
        a = embeddings[i - 1]
        b = embeddings[i]
        # Cosine similarity
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        if sim < threshold:
            boundaries.append(i)
    return boundaries


# ── Main chunker ──────────────────────────────────────────────────────────────

class SmartChunker:
    """
    Structure-aware semantic chunker inspired by SmartChunk.
    Lots of Love and credits to Ayushman Mukherjee!!!!
    Built specifically for SRag's web-scraped content pipeline.

    Pipeline:
    1. Split into sentences (structure-aware)
    2. Filter noise and boilerplate
    3. Deduplicate near-identical sentences
    4. Embed sentences (reuses SRag's existing model)
    5. Detect semantic boundaries (low similarity valleys)
    6. Pack sentences into token-budgeted chunks with overlap
    7. Emit chunk dicts with metadata (coherence score, heading path)
    """

    def __init__(
        self,
        model: Optional[SentenceTransformer] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_tokens: int = 256,
        overlap_sentences: int = 1,
        semantic_threshold: float = 0.75,
        min_chunk_chars: int = 100,
        dedupe_threshold: float = 0.85,
    ):
        # Reuse existing model if passed (avoids reloading)
        self.model = model or SentenceTransformer(model_name)
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences
        self.semantic_threshold = semantic_threshold
        self.min_chunk_chars = min_chunk_chars
        self.dedupe_threshold = dedupe_threshold

    def chunk(
        self,
        text: str,
        source_url: str = "",
        title: str = "",
    ) -> List[dict]:
        """
        Chunk a single document into semantically meaningful pieces.
        Returns list of chunk dicts ready for LanceDB indexing.
        """
        if not text or not text.strip():
            return []

        # Step 1 — Split into sentences
        sentences = _split_sentences(text)
        if not sentences:
            return []

        # Step 2 — Filter noise
        sentences = [s for s in sentences if not _is_noise(s)]
        if not sentences:
            return []

        # Step 3 — Deduplicate near-identical sentences
        deduped = []
        for s in sentences:
            if not any(_is_near_duplicate(s, existing, self.dedupe_threshold)
                       for existing in deduped[-5:]):  # check last 5 only for speed
                deduped.append(s)
        sentences = deduped

        if not sentences:
            return []

        # Step 4 — Embed all sentences
        embeddings = self.model.encode(
            sentences,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # Step 5 — Find semantic boundaries
        boundaries = _find_semantic_boundaries(
            sentences, embeddings, self.semantic_threshold
        )
        boundary_set = set(boundaries)

        # Step 6 — Pack into token-budgeted chunks
        chunks = []
        current_sentences = []
        current_tokens = 0
        chunk_index = 0

        for i, sentence in enumerate(sentences):
            s_tokens = _estimate_tokens(sentence)

            # Force split at semantic boundary OR token budget exceeded
            at_boundary = i in boundary_set
            over_budget = current_tokens + s_tokens > self.max_tokens

            if (at_boundary or over_budget) and current_sentences:
                chunk = self._make_chunk(
                    current_sentences,
                    embeddings[[sentences.index(s) for s in current_sentences
                                 if s in sentences[:len(embeddings)]]],
                    chunk_index,
                    source_url,
                    title,
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1

                # Overlap — carry last N sentences into next chunk
                current_sentences = current_sentences[-self.overlap_sentences:]
                current_tokens = sum(_estimate_tokens(s) for s in current_sentences)

            current_sentences.append(sentence)
            current_tokens += s_tokens

        # Flush remaining
        if current_sentences:
            chunk = self._make_chunk(
                current_sentences,
                embeddings[-len(current_sentences):],
                chunk_index,
                source_url,
                title,
            )
            if chunk:
                chunks.append(chunk)

        return chunks

    def chunk_docs(self, docs: List[dict]) -> List[dict]:
        """
        Chunk a list of scraped docs.
        Replaces each doc with N semantically clean chunk dicts.
        """
        all_chunks = []
        for doc in docs:
            chunks = self.chunk(
                text=doc.get("content", ""),
                source_url=doc.get("source", ""),
                title=doc.get("title", ""),
            )
            # Inherit parent metadata
            for chunk in chunks:
                chunk["timestamp"] = doc.get("timestamp", "")
                chunk["author"] = doc.get("author", "")
                chunk["image"] = doc.get("image", "")
            all_chunks.extend(chunks)

        return all_chunks

    def _make_chunk(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
        index: int,
        source_url: str,
        title: str,
    ) -> Optional[dict]:
        text = " ".join(sentences)
        if len(text.strip()) < self.min_chunk_chars:
            return None

        # Coherence score — avg adjacent cosine similarity
        coherence = self._coherence_score(embeddings)

        return {
            "source": source_url,
            "title": f"{title} [chunk {index}]",
            "content": text,
            "chunk_index": index,
            "token_estimate": _estimate_tokens(text),
            "coherence_score": round(coherence, 4),
            "sentence_count": len(sentences),
        }

    def _coherence_score(self, embeddings: np.ndarray) -> float:
        """
        Average cosine similarity between adjacent sentences.
        Higher = more topically focused chunk.
        """
        if len(embeddings) < 2:
            return 1.0
        scores = []
        for i in range(1, len(embeddings)):
            a, b = embeddings[i - 1], embeddings[i]
            sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            scores.append(float(sim))
        return sum(scores) / len(scores)