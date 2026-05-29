<div align="center">

```
███████╗██████╗  █████╗  ██████╗
██╔════╝██╔══██╗██╔══██╗██╔════╝
███████╗██████╔╝███████║██║  ███╗
╚════██║██╔══██╗██╔══██║██║   ██║
███████║██║  ██║██║  ██║╚██████╔╝
╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝
```

**Smart Retrieval-Augmented Generation**

*No API keys. No cloud costs. No data leaving your device.*

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)](LICENSE)
[![Version](https://img.shields.io/badge/version-v1.0.0-purple.svg?style=flat-square)](https://github.com/square-box-hash/SRag)
[![PyPI](https://img.shields.io/badge/pypi-srag-orange.svg?style=flat-square)](https://pypi.org/project/srag)
[![Status](https://img.shields.io/badge/status-stable-brightgreen.svg?style=flat-square)](https://github.com/square-box-hash/SRag)

</div>

---

## What is SRag?

SRag is a **local, self-hosted Python library and CLI** that replaces an entire stack of paid retrieval APIs with a single open-source package.

| What you'd normally pay for | SRag replaces it with |
|---|---|
| Tavily — search API | SearXNG + DuckDuckGo, zero cost, no key |
| Firecrawl — scraping API | httpx + trafilatura + Playwright |
| Perplexity — search + answer | Full RAG pipeline, local |
| Pinecone / Weaviate — vector DB | LanceDB, embedded, session-isolated |
| Cohere Rerank — reranking API | CrossEncoder, runs locally |

Everything runs on your machine. Queries never pass through a corporate proxy or third-party server. What happens, happens on your device.

---

## Why SRag is different

Most RAG tools are either too simple (just a wrapper around one API) or too complex (distributed infrastructure for production scale). SRag sits in the middle — **a complete, intelligent retrieval pipeline** that runs entirely locally.

**It learns.** SRag builds a topic-aware lexicon from your searches — tracking which terms appear in which domains, which sources are reliable, and which queries need deeper retrieval. It gets smarter the more you use it.

**It adapts.** Adaptive concurrency automatically tunes parallelism based on observed latency. Domain reputation scores update after every search. Query intelligence injects context from previous sessions into follow-up searches.

**It's modular.** Every subsystem — the reranker, Playwright fallback, lexicon, reputation store, quality evaluator — can be toggled on or off via `SRagConfig`. Run a lightweight version with no reranker and no Playwright when RAM is tight. Run the full stack when quality matters.

**It's typed.** Every public method returns a typed `SRagResult` with `.success`, `.chunks`, `.sources`, `.trace`, and output methods like `.to_prompt()`, `.to_json()`, `.to_mongodb()`. IDE-friendly, agent-ready.

---

## Installation

```bash
# PyPI — stable release
pip install srag

# With PDF and DOCX support
pip install srag[docs]

# With PostgreSQL support
pip install srag[databases]

# Everything
pip install srag[all]

# Latest from GitHub
pip install git+https://github.com/square-box-hash/SRag.git@v1.0.0

# Local development
git clone https://github.com/square-box-hash/SRag.git
cd SRag
pip install -e .
```

**Setup**

```bash
cp .env.example .env
# Add your HuggingFace token (free):
# HF_TOKEN=your_token_here
```

---

## Quickstart

```python
from srag import SRag

sr = SRag()

# Search and index
result = await sr.search("West Bengal election results 2026", session="wb_elections")

# Query the indexed session
chunks = sr.query("who won the most seats?", session="wb_elections", k=5)

# Build structured context for an LLM
context = sr.build_context("who won the most seats?", session="wb_elections")
print(context.to_prompt())

# Export for downstream systems
print(result.to_json())        # clean JSON
print(result.to_mongodb())     # MongoDB-ready document
print(result.to_jsonl())       # JSONL for streaming pipelines
print(result.trace.summary())  # fetch=210ms chunk=45ms embed=380ms rerank=90ms
```

**CLI in 30 seconds**

```bash
srag search "Suvendu Adhikari West Bengal"
srag read 3                                      # read result #3
srag index "FastAPI tutorial" --session fastapi
srag query "how do I add middleware?" --session fastapi
srag ingest report.pdf --session research        # local file ingestion
srag inspect --show candidates                   # see what SRag is learning
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                              │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SRagOrchestrator                           │
│                                                                 │
│  ┌─────────────────┐    ┌──────────────────┐                    │
│  │ TopicClassifier │    │ QueryIntelligence|                    │
│  │ tech/finance/   │    │ context injection│                    │
│  │ sports/news/... │    │ query expansion  │                    │
│  └────────┬────────┘    └────────┬─────────┘                    │
│           └────────────┬─────────┘                              │
└────────────────────────┼────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SearchLayer (v1.0.0)                       │
│                                                                 │
│  Priority 1: SearXNG  ←─── self-hosted or public instance       │
│       ↓ fallback if unavailable                                 │
│  Priority 2: DuckDuckGo DDGS  ←─── always available             │
│       ↓                                                         │
│  URL dedup + ReputationAwareSelector  →  ranked URL list        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   AnuInfrastructureScraper                      │
│                                                                 │
│  httpx async (parallel fetching, retry + backoff)               │
│         ↓                                                       │
│  Playwright fallback  ←─── JS-heavy pages                       │
│         ↓                                                       │
│  trafilatura / BeautifulSoup4  ←─── content extraction          │
│         ↓                                                       │
│  content hash dedup  ←─── document-level deduplication          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                       SmartChunker                              │
│                                                                 │
│  sentence splitting  →  noise filtering  →  sentence dedup      │
│         ↓                                                       │
│  sentence-transformers embed  (all-MiniLM-L6-v2, 384d)          │
│         ↓                                                       │
│  semantic boundary detection  ←─── cosine similarity valleys    │
│         ↓                                                       │
│  token-budgeted chunk packing  (max 256 tokens, 1 overlap)      │
│         ↓                                                       │
│  coherence scoring  ←─── avg adjacent cosine similarity         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                RecencyRanker + QualityEvaluator                 │
│                                                                 │
│  recency_score × recency_weight + coherence × (1-weight)        │
│  pass_rate < 0.25 → quality gate rejects session                │
│  passed_chunks < 3 → quality gate rejects session               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                       SRagIndexer                               │
│                                                                 │
│  LanceDB  ←─── session-isolated vector tables                   │
│  metadata: coherence, chunk_index, token_estimate, timestamp    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RETRIEVAL                                   │
│                                                                 │
│  query_session()  →  ANN vector search  →  candidates           │
│         ↓                                                       │
│  CrossEncoder reranker  (ms-marco-MiniLM-L-6-v2)                │
│         ↓                                                       │
│  ContextBuilder  →  token-budgeted context  →  LLM prompt       │
│         ↓                                                       │
│  SRagResult  →  .to_prompt() / .to_json() / .to_mongodb()       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  INTELLIGENCE LAYER (persistent)                │
│                                                                 │
│  ReputationStore  ←─── per-domain, per-topic confidence scores  │
│  LexiconStore     ←─── self-building query vocabulary           │
│  SideChannelCollector ←─── latency, hit rate, quality signals   │
│  AdaptiveConcurrency  ←─── dynamic parallelism tuning           │
│  SRagTracer       ←─── per-step timing attached to results      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Search providers — SearXNG + DDGS (v1.0.0)

SRag v1.0.0 introduces a priority-based search layer. SearXNG is tried first — it aggregates Google, Bing, Brave, and 70+ engines simultaneously with no rate limits. DDGS fills the gap if SearXNG is unavailable.

```python
from srag import SRag, SRagConfig

# Self-hosted SearXNG instance (recommended)
sr = SRag(config=SRagConfig(
    searxng_instance = "http://localhost:8080",
    searxng_engines  = ["google", "bing", "brave", "duckduckgo"],
))

# Auto-detect public SearXNG instance + DDGS fallback
sr = SRag(config=SRagConfig(
    use_searxng = True,
    use_ddgs    = True,
))

# DDGS only — original behavior
sr = SRag(config=SRagConfig(
    use_searxng = False,
    use_ddgs    = True,
))
```

---

## Feature toggles — `SRagConfig`

Every subsystem can be toggled independently. Use presets or build your own:

```python
from srag import SRag, SRagConfig

# Lightweight — no reranker, no Playwright, minimal RAM
sr = SRag(config=SRagConfig.lightweight())

# Fast — skip heavy steps, fewer results
sr = SRag(config=SRagConfig.fast())

# Full — everything on, maximum quality (default)
sr = SRag(config=SRagConfig.full())

# Custom — fine-grained control
sr = SRag(config=SRagConfig(
    use_reranker            = False,   # skip CrossEncoder
    use_playwright          = False,   # skip headless browser
    use_lexicon             = True,    # keep learning
    use_reputation          = True,    # keep domain scoring
    use_quality_evaluator   = True,    # keep quality gate
    use_recency_ranking     = True,    # boost recent sources
    use_searxng             = True,    # SearXNG primary search
    recency_weight          = 0.4,     # 40% recency, 60% coherence
    max_results             = 8,
    chunk_size              = 128,
    trace_timing            = True,    # enable per-step timing
))
```

---

## Typed results — `SRagResult`

Every search operation returns a typed `SRagResult`:

```python
result = await sr.search("python asyncio", session="python")

# Check success
if result.success:
    print(result.doc_count)      # documents scraped
    print(result.chunk_count)    # chunks created
    print(result.topic)          # detected topic

# Timing trace
print(result.trace.summary())
# fetch=210ms chunk=45ms embed=380ms rerank=90ms docs=8 chunks=34

# Output formats
print(result.to_prompt())        # LLM-ready context string
print(result.to_json())          # clean JSON
print(result.to_jsonl())         # JSONL for streaming pipelines

# MongoDB — drop directly into a collection
collection.insert_one(result.to_mongodb())
```

---

## Search modes

```python
# Single search — one query, one session
result = await sr.search("GST rate India 2026", session="gst")

# Parallel search — multiple independent queries concurrently
results = await sr.parallel_search([
    {"query": "FastAPI performance", "session": "fastapi"},
    {"query": "Django performance",  "session": "django"},
])

# Sequential search — step 2 informed by step 1
results = await sr.sequential_search([
    {"query": "GST rate India electronics",  "session": "gst_base"},
    {"query": "GST filing deadline 2026",    "session": "gst_deadline",
     "depends_on": "gst_base"},
])

# Verification search — conflict detection across sources
result = await sr.verify("GST rate on laptops", session="gst_verify")
if result.status == "conflict_detected":
    print(result.conflicts)
```

---

## Local file and database ingestion

SRag ingests local files and databases using the same pipeline as web retrieval — chunked, embedded, and indexed into a session.

```python
# Single file
sr.ingest("report.pdf",       session="research")
sr.ingest("notes.txt",        session="notes")
sr.ingest("data.csv",         session="dataset")
sr.ingest("config.json",      session="config")
sr.ingest("proposal.docx",    session="proposals")

# Entire folder — ingests all supported files recursively
sr.ingest("./docs/",          session="project_docs")

# SQLite database
sr.ingest("sqlite:///mydb.sqlite", session="db", table="articles")
sr.ingest("sqlite:///mydb.sqlite", session="db",
          query="SELECT title, body FROM posts WHERE published=1")

# PostgreSQL
sr.ingest("postgresql://user:pass@localhost/mydb", session="pg",
          query="SELECT * FROM articles ORDER BY created_at DESC LIMIT 1000")
```

**Supported formats**

| Format | Notes |
|--------|-------|
| `.pdf` | Text layer only, no OCR. Install: `pip install srag[docs]` |
| `.docx` | Full paragraph extraction. Install: `pip install srag[docs]` |
| `.txt` | Plain UTF-8 text |
| `.csv` | Each row becomes a chunk via pandas |
| `.json` | Flattened key-value chunks |
| SQLite | Via `sqlite://` URI |
| PostgreSQL | Via `postgresql://` URI. Install: `pip install srag[databases]` |

---

## CLI reference

```bash
# Search
srag search "query"                              # live web search
srag search "query" --results 20 --debug        # more results, debug + timing

# Read
srag read <url>                                  # fetch and read any URL
srag read 3                                      # read result #3 from last search
srag read report.pdf                             # read a local file

# Index and query
srag index "query" --session name               # search + index to session
srag index "query" --session name --force-new   # wipe and rebuild session
srag query "question" --session name            # semantic search in session
srag query "question" --session name --k 10     # top 10 results

# Local ingestion
srag ingest report.pdf --session research
srag ingest ./docs/ --session project
srag ingest sqlite:///db.sqlite --session db --table articles

# Verify and inspect
srag verify "claim" --session name              # conflict detection
srag inspect --show all                         # full system state
srag inspect --show reputation                  # domain reputation scores
srag inspect --show lexicon                     # active learned terms
srag inspect --show candidates                  # terms approaching graduation
srag inspect --show chunks --session name       # chunks in a session

# Session management
srag sessions                                    # list all sessions
srag stale session_name --hours 48              # check if session is stale
```

---

## Inspect — see what SRag is learning

```
  ════════════════════════════════════════════════════════════
  SRag Inspector
  ════════════════════════════════════════════════════════════

  LEXICON
  ────────────────────────────────────────────────────────────
  Active: 0  Candidate: 47  Suppressed: 0

  Term                     Topic          Obs  AvgConf
  ──────────────────────── ──────────── ─────  ───────
  🔵 rate                   finance         23    0.659
  🔵 mobile phones          finance         18    0.665
  🔵 fastapi dependency     tech             4    0.680
  🔵 dependency injection   tech             4    0.680
  🔵 championship standings sports           4    0.617
```

The lexicon graduates terms from candidate → active after 8+ high-confidence observations. Active terms are used for query expansion — automatically broadening searches with known related vocabulary.

---

## Error handling

```python
from srag import SRag, SRagSessionNotFoundError, SRagQualityError

sr = SRag()

try:
    chunks = sr.query("question", session="nonexistent")
except SRagSessionNotFoundError as e:
    print(e.message)   # "Session 'nonexistent' not found. Run `srag index` first."

# All exceptions
from srag import (
    SRagFetchError,         # URL fetch failed
    SRagTimeoutError,       # fetch timed out
    SRagBlockedError,       # domain blocked request
    SRagQualityError,       # quality gate rejected session
    SRagNoContentError,     # no usable content found
    SRagIndexError,         # LanceDB indexing failed
    SRagIngestError,        # local file ingestion failed
    SRagUnsupportedFormatError,   # unsupported file type
    SRagMissingDependencyError,   # optional dep not installed
)
```

---

## Roadmap

| Version | Status | Focus |
|---------|--------|-------|
| v0.4.0 | ✅ shipped | Core pipeline — search, scrape, index, CLI |
| v0.5.0 | ✅ shipped | Parallel + sequential + verification search, unified API |
| v0.6.0 | ✅ shipped | Playwright fallback, semantic chunking, query expansion |
| v0.7.0 | ✅ shipped | CrossEncoder reranker, caching, domain-aware search |
| v0.8.0 | ✅ shipped | Topic-aware lexicon, coherence scoring, `srag read`, `srag inspect`, result caching |
| v0.9.0 | ✅ shipped | Local file ingestion (PDF/DOCX/TXT/CSV/JSON), database ingestion, `SRagConfig` feature toggles, recency-aware ranking |
| v1.0.0 | ✅ shipped | PyPI publish, SearXNG+DDGS search layer, `SRagResult` typed returns, `SRagTracer` timing, typed exceptions, stable public API |

---

## Requirements

- Python 3.10+
- HuggingFace token (free) for model downloads
- Optional: `pip install srag[docs]` for PDF/DOCX
- Optional: `pip install srag[databases]` for PostgreSQL

---

## License

MIT — see [LICENSE](LICENSE)

---

<div align="center">

*Built as the retrieval layer for [Anu](https://github.com/square-box-hash) — a multi-specialist adaptive AI routing architecture.*

*Built by a class 11 student who got tired of paying for retrieval APIs.*

</div>