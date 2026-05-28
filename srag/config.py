from dataclasses import dataclass, field

@dataclass
class SRagConfig:
    # ── Feature toggles ───────────────────────────────────────
    use_reranker:              bool = True
    use_playwright:            bool = True
    use_lexicon:               bool = True
    use_reputation:            bool = True
    use_adaptive_concurrency:  bool = True
    use_quality_evaluator:     bool = True
    use_query_intelligence:    bool = True
    use_dedup:                 bool = True

    # ── Search provider settings ──────────────────────────────
    use_searxng:           bool      = True
    use_ddgs:              bool      = True   # always keep as fallback
    searxng_instance:      str       = ""     # empty = auto-detect public instance
    searxng_engines:       list      = None   # None = use default engine list

    # ── Retrieval settings ────────────────────────────────────
    max_results:       int   = 12
    max_chars:         int   = 2000
    extract_mode:      str   = "trafilatura"
    max_concurrent:    int   = 5
    rerank_top_k:      int   = 5
    chunk_size:        int   = 256
    dedupe_threshold:  float = 0.85
    db_path:           str   = "./srag_db"

    # ── Recency settings ──────────────────────────────────────
    use_recency_ranking:  bool  = True
    recency_weight:       float = 0.4   # 0.0 = pure coherence, 1.0 = pure recency
    recency_decay_days:   int   = 365

    # ── Trace/debug settings ──────────────────────────────────
    trace_timing:    bool = False   # per-step timing info
    trace_log_level: str  = "INFO"

    # ── Ingestor settings ─────────────────────────────────────
    ingestor_max_pdf_pages:  int = 500
    ingestor_csv_max_rows:   int = 10000
    ingestor_chunk_size:     int = 256

    # ── Presets ───────────────────────────────────────────────
    @classmethod
    def lightweight(cls) -> "SRagConfig":
        """Minimal RAM — no reranker, no playwright, no lexicon."""
        return cls(
            use_reranker             = False,
            use_playwright           = False,
            use_lexicon              = False,
            use_adaptive_concurrency = False,
            use_quality_evaluator    = False,
            use_query_intelligence   = False,
        )

    @classmethod
    def fast(cls) -> "SRagConfig":
        """Speed over quality — skip heavy steps."""
        return cls(
            use_reranker   = False,
            use_playwright = False,
            max_results    = 6,
            rerank_top_k   = 3,
        )

    @classmethod
    def full(cls) -> "SRagConfig":
        """Everything on — maximum quality."""
        return cls()