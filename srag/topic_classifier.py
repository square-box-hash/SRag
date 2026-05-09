import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Topic keyword map ─────────────────────────────────────────────────────────

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "tech": [
        "python", "api", "code", "github", "library", "framework", "model",
        "llm", "ai", "machine learning", "deep learning", "neural network",
        "dataset", "training", "inference", "deployment", "docker", "cloud",
        "backend", "frontend", "devops", "compiler", "algorithm", "tutorial",
        "documentation", "opensource", "version control", "debugging", "performance", "scalability",
        "security", "vulnerability", "exploit", "patch", "update", "release",
        "benchmark", "evaluation", "research", "innovation", "startup", "funding",
        "investment", "acquisition", "merger", "layoff", "hiring", "job market",
        "remote work", "hybrid work", "office culture", "diversity", "inclusion",
    ],
    "finance": [
        "gst", "tax", "income tax", "rate", "market", "stock", "equity",
        "shares", "ipo", "investment", "mutual fund", "sip", "rbi", "inflation",
        "interest rate", "banking", "budget", "fiscal policy", "revenue",
        "profit", "loss", "cryptocurrency", "bitcoin", "trading", "forex", "currency",
        "economy", "economic growth", "recession", "unemployment", "gdp",
        "fiscal stimulus", "monetary policy", "quantitative easing", "deficit",
        "debt", "credit", "rating", "dow jones", "s&p 500", "nasdaq",
        "sensex", "nifty", "bse", "nse", "commodity", "gold", "silver",
        "oil", "gas", "real estate", "housing market", "mortgage", "property",
        "realestate", "realestate market", "realestate development", "realestate investment",
        "realestate consultancy", "realestate broker", "realestate agent", "realestate broker",
    ],
    "science": [
        "research", "study", "paper", "experiment", "physics", "chemistry",
        "biology", "mathematics", "theorem", "equation", "data analysis",
        "simulation", "astronomy", "genetics", "neuroscience", "lab",
        "hypothesis", "scientific method", "discovery", "innovation", "nobel", "award", "conference",
        "publication", "peer review", "replication", "open science", "science communication",
        "climate change", "environment", "sustainability", "energy", "renewable energy", "fossil fuels", "carbon emissions", "biodiversity", "conservation",
        "space exploration", "space mission", "mars", "moon", "satellite", "telescope",
        "quantum mechanics", "relativity", "string theory", "dark matter", "dark energy", "multiverse", "cosmology", "particle physics", "higgs boson", 
        "gravitational waves", "black hole", "wormhole", "time travel", "parallel universe", "extraterrestrial life", "astrobiology", "exoplanet", "space colonization",
        "artificial intelligence", "machine learning", "deep learning", "neural network", "neural network", "neural network", "neural network", "neural network", "neural network",
    ],
    "sports": [
        "cricket", "football", "soccer", "basketball", "tennis", "badminton",
        "olympics", "tournament", "match", "score", "result", "league", "ipl",
        "world cup", "fifa", "f1", "gp", "formula 1", "driver", "team",
        "standings", "highlights", "wins", "race", "winner", "record", "podium",
        "championship", "champion", "athlete", "player", "coach", "manager",
        "referee", "grand prix", "qualifying", "pit stop", "overtake", "safety car",
        "penalty", "yellow flag", "red flag", "lap time", "fastest lap", "pole position",
        "driver of the day", "constructor", "engine", "aerodynamics", "tire strategy", "weather conditions",
    ],
    "news": [
        "breaking", "headline", "report", "update", "government", "policy",
        "election", "minister", "parliament", "law", "court", "verdict",
        "international", "geopolitics", "conflict", "economy news", "regulation", 
        "scandal", "investigation", "protest", "diplomacy", "summit", "treaty",
        "sanction", "aid", "humanitarian", "refugee", "climate change", "environment", "disaster", "corruption", "whistleblower",
    ],
    "education": [
        "exam", "syllabus", "cbse", "icse", "jee", "neet", "university",
        "admission", "course", "lecture", "notes", "assignment", "result",
        "scholarship", "coaching", "preparation", "study material",
        "tutor", "online learning", "edtech", "curriculum", "grading",
        "academic", "research", "publication", "conference", "workshop", "seminar",
    ],
    "entertainment": [
        "movie", "film", "trailer", "actor", "actress", "music", "song",
        "album", "series", "netflix", "bollywood", "hollywood", "celebrity",
        "review", "box office", "release", "trailer", "song", "lyrics",
        "songwriter", "composer", "lyricist", "musician", "music video",
        "pop music", "rock music", "hip hop music", "electronic music",
        "classical music", "jazz music", "country music", "reggae music",
        "blues music", "folk music", "world music", "indie music", "soul music",
    ],
    "health": [
        "disease", "symptoms", "treatment", "medicine", "hospital", "doctor",
        "fitness", "diet", "nutrition", "mental health", "exercise", "vaccine",
        "diagnosis", "healthcare", "wellness", "public health", "epidemic", "pandemic", "virus",
        "bacteria", "immunity", "health policy", "health insurance", "medical research", "clinical trial", "pharmaceutical", "side effect", "health guideline",
    ],
    "general": [
        "news", "sports", "finance", "tech", "science", "health",
    ],
}

AMBIGUITY_THRESHOLD = 0.15


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class TopicResult:
    primary:              str
    confidence:           float
    secondary:            Optional[str]
    secondary_confidence: float
    ambiguous:            bool
    top_k:                list[tuple[str, float]]   # all topics with score > 0, sorted

    def __repr__(self) -> str:
        amb = " [AMBIGUOUS]" if self.ambiguous else ""
        return (
            f"TopicResult(primary={self.primary!r} ({self.confidence:.2f})"
            f", secondary={self.secondary!r} ({self.secondary_confidence:.2f})"
            f"{amb})"
        )


# ── Classifier ────────────────────────────────────────────────────────────────

class TopicClassifier:
    """
    Keyword-based topic classifier for search queries.

    Scores each topic by counting keyword hits (unigrams and bigrams),
    normalizes scores, and returns a TopicResult with primary/secondary
    topics and an ambiguity flag.

    Optionally accepts active lexicon terms from LexiconStore to augment
    the static keyword map with learned terms, weighted by term_weight.
    """

    def __init__(self, lexicon_boost: float = 0.5):
        """
        Args:
            lexicon_boost: weight multiplier applied to lexicon term hits
                           relative to static keyword hits (default 0.5
                           so learned terms don't dominate static ones)
        """
        self.lexicon_boost = lexicon_boost

    def predict(
        self,
        query: str,
        top_k: int = 2,
        lexicon_terms: Optional[dict[str, list[dict]]] = None,
    ) -> TopicResult:
        """
        Classify query into topic(s).

        Args:
            query:         search query string
            top_k:         number of topics to return in TopicResult.top_k
            lexicon_terms: optional dict of {topic: [{"term": str, "weight": float}]}
                           from LexiconStore.get_active_terms() — augments scoring

        Returns:
            TopicResult with primary, secondary, confidence, ambiguity flag,
            and full top_k ranking.
        """
        if not query or not query.strip():
            return self._fallback(top_k)

        q           = query.lower().strip()
        raw_scores  = self._score_static(q)

        if lexicon_terms:
            raw_scores = self._score_lexicon(q, raw_scores, lexicon_terms)

        if not any(s > 0 for s in raw_scores.values()):
            return self._fallback(top_k)

        # Normalize scores to [0, 1]
        total  = sum(raw_scores.values())
        normed = {
            t: round(s / total, 4)
            for t, s in raw_scores.items()
            if s > 0
        }

        sorted_topics = sorted(normed.items(), key=lambda x: x[1], reverse=True)
        top_k_result  = sorted_topics[:max(top_k, 2)]

        primary,   primary_conf   = top_k_result[0]
        secondary, secondary_conf = top_k_result[1] if len(top_k_result) > 1 else (None, 0.0)

        ambiguous = (
            secondary is not None and
            abs(primary_conf - secondary_conf) < AMBIGUITY_THRESHOLD
        )

        logger.debug(
            "TopicClassifier: query=%r primary=%s (%.2f) secondary=%s (%.2f) ambiguous=%s",
            query, primary, primary_conf, secondary, secondary_conf, ambiguous,
        )

        return TopicResult(
            primary              = primary,
            confidence           = primary_conf,
            secondary            = secondary,
            secondary_confidence = secondary_conf,
            ambiguous            = ambiguous,
            top_k                = top_k_result,
        )

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_static(self, q: str) -> dict[str, float]:
        """Score topics using static keyword map. Multi-word keywords handled."""
        scores: dict[str, float] = {topic: 0.0 for topic in TOPIC_KEYWORDS}
        for topic, keywords in TOPIC_KEYWORDS.items():
            for kw in keywords:
                if kw in q:
                    # Multi-word keywords score higher than single words
                    scores[topic] += 1.0 + (0.2 * (kw.count(" ")))
        return scores

    def _score_lexicon(
        self,
        q: str,
        base_scores: dict[str, float],
        lexicon_terms: dict[str, list[dict]],
    ) -> dict[str, float]:
        """
        Augment base scores with active lexicon terms.
        Lexicon hits are weighted by term_weight * lexicon_boost.
        """
        scores = dict(base_scores)
        for topic, terms in lexicon_terms.items():
            if topic not in scores:
                scores[topic] = 0.0
            for term_record in terms:
                term   = term_record.get("term", "")
                weight = float(term_record.get("weight", 0.0))
                if term and term in q:
                    scores[topic] += weight * self.lexicon_boost
        return scores

    # ── Fallback ──────────────────────────────────────────────────────────────

    def _fallback(self, top_k: int) -> TopicResult:
        """Returns general topic when no keywords match."""
        logger.debug("TopicClassifier: no keywords matched, falling back to 'general'")
        return TopicResult(
            primary              = "general",
            confidence           = 1.0,
            secondary            = None,
            secondary_confidence = 0.0,
            ambiguous            = False,
            top_k                = [("general", 1.0)][:top_k],
        )