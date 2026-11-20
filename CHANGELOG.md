# Changelog

All notable changes to SRag are documented here.
Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Semantic Versioning](https://semver.org/).

## [1.3.10] — 2026-11-20

### Fixed
- **The container served no MCP endpoint.** `srag[all]` never included the `mcp`
  extra and the Dockerfile installed exactly that, so an image built from it
  answered HTTP only and an agent had nothing to connect to. `all` includes `mcp`
  now, and the image asks for it explicitly in case the extra list changes again.
- **The image was unpinned.** It installed whatever `srag` PyPI served at build
  time — including 1.3.8, whose `server.py` does not parse. It pins the release.
- **The volume was mounted where nothing wrote.** `db_path` defaults to
  `./srag_db` while compose mounted `/app/srag_data`, so every restart dropped the
  index and re-searched from scratch. `SRagConfig.db_path` now honours
  `SRAG_DB_PATH` and `SRAG_DATA_DIR`, and the explicit `"./srag_db"` defaults in
  `SRag` and `SRagOrchestrator` no longer override the configured path. The
  compose file mounts at the default path, so it works on this release either way.
- **The healthcheck did not warm the model.** It called `/health`, which never
  builds the SRag instance, so the container reported healthy while the indexer
  and embedding model were still cold and the first search paid that cost. It
  calls `/sessions` now, with the bearer header when a token is set.

### Changed
- The container publishes on `127.0.0.1` instead of every interface, and compose
  passes `SRAG_API_TOKEN`, `SRAG_INGEST_ROOTS` and `SRAG_ALLOWED_ORIGINS`. A
  service behind a tunnel is reachable by anyone who learns the hostname, and
  tunnel hostnames are published to certificate transparency logs — the token is
  the authentication, the URL is not.
- README documents running SRag as a container behind a public URL, and states
  plainly that a tunnel is reachability rather than hosting: the URL forwards to a
  bridge on the machine that started it, so that machine has to stay on.

### Added
- `.env.example` — the README had told people to copy one that did not exist.

## [1.3.9] — 2026-11-20

### Fixed
- **`srag/server.py` did not parse, so the HTTP and MCP server could not start.**
  Two entries in the `endpoints` map inside `root()` had lost their colons
  (`"read" "POST /read"`), which Python reads as implicit string concatenation
  and then rejects with `SyntaxError: invalid syntax` at import. Both `srag serve`
  and `uvicorn srag.server:app` died before binding a port. The 1.3.8 wheel on
  PyPI contained the same bytes, so this affected every install, not just `main`.
- **`srag[mcp]` produced a server with no MCP endpoint.** `fastapi-mcp` 0.4.x
  calls `mcp.Server(name, version)` positionally, and the `mcp` 2.x SDK changed
  that signature, so the mount raised `TypeError`. `_mount_mcp()` caught only
  `ImportError`, so the exception escaped module import and took the whole app
  down. The extra now pins `mcp>=1.10,<2`, and any mount failure is logged and
  degraded instead of fatal — the HTTP API stays up and `/` reports
  `mcp.available: false` with the reason.
- `_mount_mcp()` used `mcp.mount()`, deprecated in `fastapi-mcp` 0.4. It now
  mounts `/mcp` (streamable HTTP) and `/mcp/sse` (SSE), falling back to the
  legacy call for older `fastapi-mcp`.
- **`DELETE /sessions/{name}` returned 200 and deleted nothing** — it returned a
  message telling you to POST `/search` with `force_new` instead. It now deletes
  through the new `SRag.delete_session()` / `SRagIndexer.delete_session()`, which
  drops the namespaced table and its `index_meta.json` entry, answers 404 for an
  unknown session and 501 on a build without the method.
- **Error responses echoed exception text.** All 13 `detail=str(e)` sites now log
  the real error and return a generic message, because exception text carries
  filesystem paths and connection strings with credentials.
- `tests/test_indexer_embedding_namespaces.py` hardcoded an absolute path and
  registered a fake `srag` module in `sys.modules`, which broke every other test
  module's import; it also called `sys.exit()` at import time, so pytest died
  with an INTERNALERROR instead of running it. It is now three pytest tests
  (16 → 30 in the suite) that use `tmp_path` and assert instead of printing.

### Security
- **Local file disclosure via wildcard CORS.** The server ran with
  `allow_origins=["*"]`, no authentication and a `0.0.0.0` bind. Any web page a
  user visited could `POST /ingest` with a path such as `/home/user/notes.txt`
  and then `POST /query` to read the chunks back — no auth, no user interaction
  beyond loading the page. The same origin could drive `/read` at internal hosts
  and `/search` against the user's machine.
- **`srag serve` now binds `127.0.0.1`** (was `0.0.0.0`). Pass `--host 0.0.0.0`
  deliberately to expose it on a network. *Breaking for anyone relying on the old
  default.*
- **CORS is an allowlist.** Default is local dev origins; cross-origin browser
  clients must be named in `SRAG_ALLOWED_ORIGINS`. Setting it to `*` restores the
  old behaviour explicitly. *Breaking for browser clients on other origins.*
- `SRAG_API_TOKEN` enables bearer authentication (`/health` stays open for
  liveness probes). Unset means no auth, as before.
- `SRAG_INGEST_ROOTS` confines `/ingest` to listed directories. Unset means
  unchanged behaviour.

### Added
- `tests/test_server_security.py` — 14 tests covering the CORS allowlist, the
  bearer token, the ingest roots guard, error-detail redaction and session
  deletion, including that a 401 still carries CORS headers.
- `/` now reports both `http_endpoint` and `sse_endpoint`, and the MCP tool
  surface excludes `System`-tagged routes: `/health`, `/` and `/inspect` are for
  humans and uptime checks, and every tool costs an agent prompt budget on every
  turn. 14 tools → 11.

## [1.3.8] — 2026-11-18

### Fixed
- **Dates were fabricated.** `_extract_date()` returns `None` when a page has no
  date metadata, and both call sites fell back to `datetime.utcnow()`. Every
  undated page was therefore stored as published *today*, which silently disabled
  recency ranking (`recency_weight=0.4`) and displayed a false date in the CLI.
  `timestamp` is now the published date or an empty string, and the fetch time is
  kept separately as `fetched_at`. `recency_score()` already treats an empty
  timestamp as neutral (0.5), so ranking is honest again.
- `srag read` imported `SentenceTransformer` and `SmartChunker` without ever using
  them, so the command died with ImportError on a torch-free install.

### Added
- `search(..., check_conflicts=True)` reports sources that disagree. Detection now
  runs on documents that were already fetched, so `verify()` no longer pays for a
  second search to reach the same conclusion. `SRagResult.conflicts` carries the
  list, and `srag index --check-conflicts` prints it.

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


## [1.1.0] — 2026-06-06

### Added
- **FastAPI server** (`srag/server.py`) — search, parallel and sequential search,
  verification, query, context, ingest and session listing over HTTP.
- **MCP server**, mounted from the same FastAPI app, so an agent can use SRag as a
  tool server without an adapter (`pip install srag[mcp]`).
- **`Dockerfile` and `docker-compose.yml`** for running SRag as a service, with an
  optional self-hosted SearXNG profile.
- `srag serve` on the CLI.
- A PyPI publish workflow, so a tag ships the release.

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
