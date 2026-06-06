# Changelog

All notable changes to SRag are documented here.
Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Semantic Versioning](https://semver.org/).

---

## [1.0.0] — 2026-05-28

### Added
- `SRagResult` typed return dataclass — replaces raw dicts across all public methods
- `SRagTrace` — per-step timing (fetch, chunk, embed, rerank) attached to every result
- `SRagTracer` — context-manager based pipeline tracer, zero overhead when disabled
- `RecencyRanker` — topic-aware recency scoring blended with coherence
- Typed exception hierarchy — `SRagFetchError`, `SRagQualityError`, `SRagIndexError`, `SRagSessionNotFoundError`, `SRagIngestError`, `SRagUnsupportedFormatError`, `SRagMissingDependencyError`
- `SRagResult.to_mongodb()` — drop-in MongoDB document
- `SRagResult.to_jsonl()` — JSONL output for streaming pipelines
- `SRagResult.to_json()` — clean JSON string
- `SRagResult.to_prompt()` — LLM-ready context string
- `py.typed` marker — PEP 561 compliance for type checkers
- `SearchLayer` — priority-based search with SearXNG primary + DDGS fallback
- `SearXNGProvider` — supports self-hosted and public SearXNG instances with auto-failover
- `DDGSProvider` — wrapped as a proper provider class
- `SRagConfig.searxng_instance` — point to your own SearXNG instance
- `SRagConfig.searxng_engines` — configure which engines SearXNG aggregates
- Public SearXNG instance list with automatic failover when no instance configured
- PyPI publish via GitHub Actions on version tag
- `__version__ = "1.0.0"` exported from package root
- `authors`, `keywords`, `classifiers`, `project.urls` in `pyproject.toml`

### Changed
- `orchestrator.search()` now returns `SRagResult` instead of raw dict
- `orchestrator.query()` raises `SRagSessionNotFoundError` instead of returning empty list
- Recency ranking wired into search pipeline after chunking
- Tracer wired into search pipeline, activated by `debug=True` or `config.trace_timing=True`

### Fixed
- Graceful degradation when reranker is `None` — falls back to coherence ranking
- Null-safe guards on all optional subsystems (lexicon, reputation, evaluator, controller)

---

## [0.9.0] — 2026-05-28

### Added
- `DocumentIngestor` — PDF, DOCX, TXT, CSV, JSON, SQLite, PostgreSQL ingestion
- `SRagConfig` — feature toggle dataclass with `lightweight()`, `fast()`, `full()` presets
- `srag ingest` CLI command
- `srag read` now accepts local file paths
- Folder ingestion — `sr.ingest("./docs/", session="name")`
- Optional dependency extras — `srag[docs]`, `srag[databases]`, `srag[all]`
- Sentence-level cross-chunk deduplication via `seen_sentences` set

### Changed
- `SRagOrchestrator.__init__` accepts `SRagConfig` — all subsystems conditionally initialized
- `_rerank()` falls back to coherence ranking when reranker is disabled

---

## [0.8.0] — 2026-05-09

### Added
- Topic-aware lexicon — `LexiconStore` tracks recurring search terms per topic
- `srag inspect` command with `--show {all,reputation,lexicon,candidates,chunks,collector}`
- `srag read <url|index>` — fetch and render any URL in terminal
- Result caching — `srag read 3` reads result #3 from last search
- Coherence scoring visible in `srag inspect`
- Candidate term graduation system (8+ observations → active)

---

## [0.7.0] — 2026-04-XX

### Added
- CrossEncoder reranker — ms-marco-MiniLM-L-6-v2
- Domain reputation store — per-domain, per-topic confidence scoring
- Source validation and blocked domain filtering
- `SideChannelCollector` — latency and quality signal collection

---

## [0.6.0] — 2026-03-XX

### Added
- Playwright fallback for JS-heavy pages
- `SmartChunker` — semantic chunking at topic boundaries
- Query expansion — automatic year + synonym rephrasing
- `QualityEvaluator` — session-level quality gate

---

## [0.5.0] — 2026-02-XX

### Added
- Parallel search — multiple independent queries concurrently
- Sequential search — chained queries with context injection
- Verification search — conflict detection across sources
- Unified `SRag` public API class

---

## [0.4.0] — 2026-01-XX

### Added
- Core pipeline — DuckDuckGo search, httpx scraping, LanceDB indexing
- `srag` CLI with `search`, `index`, `query`, `verify`, `sessions`, `stale`
- Session isolation — each topic gets its own LanceDB table
- Sentence-transformers embedding (all-MiniLM-L6-v2)# v1.1.0 Release
