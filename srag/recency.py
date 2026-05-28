# srag/recency.py
"""
Recency-aware ranking for SRag search results.
Boosts newer sources without completely overriding coherence scores.
"""
from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import Optional


def recency_score(
    timestamp:   str,
    decay_days:  int   = 365,
    floor:       float = 0.1,
) -> float:
    """
    Compute a recency score between 0 and 1.
    Score decays exponentially with age.
    
    Args:
        timestamp:  ISO format date string
        decay_days: Age at which score reaches ~0.37 (one decay constant)
        floor:      Minimum score for very old content
    
    Returns:
        float between floor and 1.0
    """
    if not timestamp:
        return 0.5  # neutral if no date

    try:
        pub = datetime.fromisoformat(timestamp[:10]).replace(tzinfo=timezone.utc)
        age_days = max(0, (datetime.now(timezone.utc) - pub).days)
        score = math.exp(-age_days / decay_days)
        return max(floor, score)
    except Exception:
        return 0.5


def combined_score(
    coherence:      float,
    timestamp:      str,
    recency_weight: float = 0.4,
    decay_days:     int   = 365,
) -> float:
    """
    Blend coherence and recency into a single ranking score.
    
    Args:
        coherence:      Coherence score from SmartChunker (0-1)
        timestamp:      ISO format date string
        recency_weight: 0.0 = pure coherence, 1.0 = pure recency
        decay_days:     Recency decay window
    
    Returns:
        Combined score between 0 and 1
    """
    r = recency_score(timestamp, decay_days)
    c = max(0.0, min(1.0, coherence))
    return (1 - recency_weight) * c + recency_weight * r


class RecencyRanker:
    """
    Reranks a list of chunks by combined coherence + recency score.
    Topic-aware: news queries weight recency higher than tech docs.
    """

    # Per-topic recency weights
    TOPIC_WEIGHTS: dict[str, float] = {
        "news":          0.65,
        "finance":       0.55,
        "sports":        0.60,
        "health":        0.45,
        "tech":          0.30,
        "science":       0.25,
        "education":     0.20,
        "entertainment": 0.50,
        "general":       0.40,
    }

    def __init__(
        self,
        recency_weight: float = 0.4,
        decay_days:     int   = 365,
        topic_aware:    bool  = True,
    ):
        self.recency_weight = recency_weight
        self.decay_days     = decay_days
        self.topic_aware    = topic_aware

    def _get_weight(self, topic: str) -> float:
        if self.topic_aware:
            return self.TOPIC_WEIGHTS.get(topic, self.recency_weight)
        return self.recency_weight

    def rank(
        self,
        chunks:  list[dict],
        topic:   str = "general",
        top_k:   Optional[int] = None,
    ) -> list[dict]:
        """
        Rank chunks by combined coherence + recency score.
        Adds 'combined_score' field to each chunk.
        """
        if not chunks:
            return chunks

        weight = self._get_weight(topic)

        for chunk in chunks:
            chunk["combined_score"] = combined_score(
                coherence      = chunk.get("coherence_score", 0.5),
                timestamp      = chunk.get("timestamp", ""),
                recency_weight = weight,
                decay_days     = self.decay_days,
            )

        ranked = sorted(chunks, key=lambda c: c["combined_score"], reverse=True)
        return ranked[:top_k] if top_k else ranked