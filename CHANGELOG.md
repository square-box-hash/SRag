# Changelog

All notable changes to SRag are documented here.
Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Semantic Versioning](https://semver.org/).

## [1.3.7] — 2026-11-18

### Fixed
- **`srag search` and `srag index` crashed on every run in 1.3.6.** The bulk
  rewrite that introduced `_get()` also rewrote the helper's own body into
  `return _get(...)`, so any command reaching the printers hit `RecursionError`.
  Caught by the new CLI smoke tests within a minute of writing them.

### Added
- `tests/test_public_api_contract.py` — asserts `search()`, `verify()`,
  `parallel_search()` and `sequential_search()` return `SRagResult`, that sources
  are derived and deduped, that retrieval is async with sync aliases, and that
  `ingest()` survives an indexer that returns `None`.
- `tests/test_cli_smoke.py` — drives `srag query` through `main()` with a stubbed
  `SRag`, and checks both printers against a dict *and* an `SRagResult`. These are
  the two shapes whose mismatch shipped twice.

Both suites run with no torch, no network and no LanceDB.

## [1.3.6] — 2026-11-18

### Fixed
- **`srag query` was broken by 1.3.0.** The CLI called `sr.query(...)` without
  awaiting it after `query()` became a coroutine, so it printed a coroutine object
  instead of results. It uses `query_sync()` now. Same for `srag inspect --query`.
- **`srag search` and `srag index` were broken by 1.3.4.** The CLI reads results
  with `result["success"]` and `result.get(...)`, and 1.3.4 started returning
  `SRagResult`. Both printers now accept either shape.
- `SRagResult.raw` keeps the orchestrator payload, so `--debug` output and any key
  not modelled on the dataclass survives the wrapping.

### Note
These were regressions in the library's own CLI, introduced by two of the fixes
above and invisible to the unit tests, which never invoke the CLI.

## [1.3.5] — 2026-11-18

### Fixed
- Every search logged `Reputation update failed ... 'DomainFilterStore' object has
  no attribute 'record'`. The store exposes `record_coherence()`, not
  `record.coherence()`.

  Correction to the original wording of this entry: `reputation.update()` runs
  *before* the failing line, so domain reputation **was** learning — which is why
  repeated searches for the same topic visibly improved. What the exception killed
  was everything after it: the cold-start cap (so a domain could reach high
  confidence after a single scrape) and the low-coherence flagging in
  `DomainFilterStore`.
- `trace.summary()` reported all zeros. The orchestrator never built a trace, so
  the timing fields the README advertises were always empty. `search()` now times
  fetch, chunk and index and returns a real `SRagTrace`.
- `list_sessions()` returned SRag's own bookkeeping tables (`lexicon`,
  `domain_reputation`, `domain_filters`) as if they were user sessions.

## [1.3.4] — 2026-11-18

### Fixed
- `SRag.search()` returned a raw `dict`, so every documented example —
  `result.success`, `result.chunks`, `result.trace.summary()`, `result.to_json()`
  — raised `AttributeError: 'dict' object has no attribute 'success'`. It now
  returns the `SRagResult` the README promises, with `sources` derived from the
  fetched documents. `verify()`, `parallel_search()` and `sequential_search()` are
  wrapped the same way.
- DuckDuckGo search returned nothing: the `duckduckgo_search` package was renamed
  to `ddgs`, and the old one is a dead shim. The provider now imports `ddgs`
  first, falling back to the old name, and the dependency is `ddgs>=9.0`.

## [1.3.3] — 2026-11-18

### Fixed
- `ingest()` silently dropped your own documents. The coherence filter — built to
  discard scraped web junk — defaults to 0.3, and a short note whose sentences are
  unrelated scores far below that (0.13 in the case that surfaced this), so the
  whole file was discarded and `indexed_count` came back 0.
  New `SRagConfig.ingest_min_coherence`, defaulting to **0.0**: files you supply
  are indexed. Scraped documents keep the 0.3 threshold.
- The `get_sentence_embedding_dimension` FutureWarning from
  sentence-transformers — the newer `get_embedding_dimension` is preferred, with
  the old name as a fallback.

## [1.3.2] — 2026-11-18

### Fixed
- `SRag.ingest()` crashed with `AttributeError: 'NoneType' object has no attribute
  'get'` whenever a document produced no chunks — a short note under
  `min_chunk_chars` was enough to trigger it. `index_documents()` now always
  returns a stats dict, and `ingest()` reports the real reason instead of dying.
- `ingest()` now returns `success: False` when nothing was indexed, rather than
  `True` just because chunks existed before the coherence filter.

### Added
- `SRagConfig.min_chunk_chars` (default 100), so short documents can be indexed
  without reaching into the chunker.

## [1.3.1] — 2026-11-18

### Changed
- Documentation only. PyPI renders the README frozen at upload time, so the
  1.3.0 release page still shows the pre-MCP documentation. This release
  republishes it: the MCP/HTTP section, the extras matrix, async retrieval in the
  quickstart and the real footprint.

## [1.3.0] — 2026-11-18

### Added
- **Embedding providers.** `LocalEmbedder` (sentence-transformers, offline) and
  `GoogleEmbedder` (Generative Language `batchEmbedContents` over httpx, so no new
  dependency). `SRagConfig` gains `embedding_provider`, `embedding_model`,
  `embedding_dimension` and `google_api_key`.
- **Namespaced indexes.** Tables are suffixed with provider + model + dimension,
  `index_meta.json` records what built each one, and `describe(session)` reads it
  back. Switching provider creates a sibling table instead of rewriting the old.
- **Async retrieval.** `query()`, `query_session()` and `build_context()` are now
  coroutines to match `search()`. `query_sync()`, `query_session_sync()` and
  `build_context_sync()` keep pre-1.3 code working.

### Changed
- `sentence-transformers` moved from the core dependencies into the `local`
  extra. `import srag` no longer loads torch — 154 ms, and torch, playwright and
  trafilatura stay out of `sys.modules` until something needs them.
- Heavy exports (`SRagOrchestrator`, `SRagIndexer`, `DocumentIngestor`,
  `SRagTracer`, `RecencyRanker`, `AnuInfrastructureScraper`) are resolved lazily
  through PEP 562.
- Chunking degrades to token-budget packing and the reranker is skipped when
  `sentence-transformers` is absent, instead of failing at import.

### Fixed
- A query embedded at a different dimension than the index raises
  `EmbeddingMismatchError` naming both models, rather than returning nonsense.

## [1.2.1] — 2026-07-05

### Fixed
- `from sympy import re` in `scraper.py` was shadowing the standard library
  `re` module, causing `AttributeError` on every Playwright fallback fetch
- `srag read` now falls back to Playwright if httpx gets a 403 — same
  behaviour as the main search pipeline
- `seen_sentences` dedup set in `SmartChunker` was reset per document —
  moved to `chunk_docs()` scope so duplicate content across documents
  (JS-rendered pages with duplicated DOM blocks) gets caught correctly
- `srag read` no longer instantiates `SmartChunker` and `SentenceTransformer`
  unnecessarily — raw doc content is rendered directly, eliminating the
  overlap-sentence duplication artifact in terminal output

---
## [1.2.0] — 2026-06-28

### Added
- `DOMDistiller` — structural HTML pruning pass that runs before trafilatura on every fetch
  - CSS selector blocklist strips cookie banners, nav widgets, social share bars, modals, and other noise structurally before text extraction
  - Negative-tag pruning (`nav`, `aside`, `footer`, `form`, `header`) based on link density and text length — content-bearing headers inside `<article>` are preserved
  - Per-domain selector overrides via `DomainFilterStore` layered on top of the global blocklist
  - Live-DOM Playwright path — injects a distillation script into the rendered page context before serialization, stripping computed-hidden elements (`display:none`, `visibility:hidden`, `opacity:0`) that static HTML parsing cannot see
  - Head-preservation fix for Playwright path — reconstructs `<head>` from original page content before static structural pass, so `_extract_title`/`_extract_date`/`_extract_author`/`_extract_image` continue reading `<meta>` tags correctly
- `DomainFilterStore` — per-domain content filtering store backed by LanceDB
  - Manual CSS selector overrides per domain via `set_selectors(domain, selectors)`
  - Coherence streak tracking — flags domains with 5+ consecutive low-coherence scrapes for manual review without auto-writing selectors
  - Surfaced via `srag inspect --show domain-filters`
  - `set_selectors()` clears existing flag and streak on manual fix
- `inspector.py` — shared data-gathering module used by both CLI and HTTP server
  - `inspect(db, show, session, domain, topic)` — single entry point returning structured dict
  - `get_reputation()`, `get_lexicon()`, `get_chunks()`, `get_domain_filters()` — individual data accessors
- `GET /inspect` HTTP endpoint — exposes full `srag inspect` output as JSON, query params mirror CLI flags (`show`, `session`, `domain`, `topic`)
- `POST /read` HTTP endpoint — fetches a URL through the distiller and returns clean text without indexing, HTTP equivalent of `srag read <url>`
- `SRagConfig.use_dom_distiller` feature toggle — disabled automatically in `lightweight()` and `fast()` presets
- `domain-filters` added as a valid `--show` mode for `srag inspect`

### Changed
- `srag inspect` CLI refactored — formatting logic stays in `cli.py`, all data-gathering moved to `inspector.py`; CLI and HTTP server now share the same underlying data layer
- `DomainFilterStore.record_coherence()` hooked into `_update_reputation()` in orchestrator — every domain gets coherence tracked automatically after each scrape
- `DomainFilterStore` instantiated in orchestrator alongside `ReputationStore`, passed through to scraper via `_make_scraper()`
- `SearchLayer` now properly instantiated in `SRagOrchestrator.__init__` — was referenced but never assigned, causing `AttributeError` on every `search()` call
- `self.max_concurrent` now assigned in `SRagOrchestrator.__init__` — was referenced in `parallel_search()` log line but never set

### Fixed
- `__version__` corrected to match `pyproject.toml` — was stuck at `"1.0.0"` despite v1.1.0 being tagged and published
- `srag[mcp]` extra now exists in `pyproject.toml` — `pip install "srag[mcp]"` as documented in v1.1.0 release notes previously errored with "no such extra"
- `fastapi-mcp` pinned at `>=0.3.0` (current: 0.4.0) — was undeclared as a dependency

### Dependencies
- Added `lxml>=5.0` — required by `DOMDistiller` for DOM tree parsing and manipulation
- Added `cssselect>=1.2` — required by `DOMDistiller` for CSS selector matching via `lxml`
- Added `playwright>=1.3` — required by `DOMDistiller` for live-DOM scraping


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
