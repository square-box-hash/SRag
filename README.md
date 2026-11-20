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

<sub>Local by default — that includes the embedding model. Bring a Google key only if
you would rather not run one yourself: see [Embedding providers](#embedding-providers).</sub>

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)](https://github.com/square-box-hash/SRag/blob/main/LICENSE)
[![Version](https://img.shields.io/pypi/v/srag?style=flat-square&color=purple)](https://pypi.org/project/srag/)
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

# With local embeddings + reranker (pulls torch, ~800 MB)
pip install srag[local]

# With PostgreSQL support
pip install srag[databases]

# Everything
pip install srag[all]

# Latest from GitHub (main branch)
pip install git+https://github.com/square-box-hash/SRag.git

# Or pin a specific release
pip install git+https://github.com/square-box-hash/SRag.git@v1.3.9

# Local development
git clone https://github.com/square-box-hash/SRag.git
cd SRag
pip install -e .
```

**Setup**

```bash
cp .env.example .env
```

Everything in `.env` is optional. The settings you are most likely to touch:

```bash
EMBEDDING_PROVIDER=local      # or "google" (then set GOOGLE_API_KEY)
SEARXNG_INSTANCE=             # your own SearXNG; blank = auto-detect + DDGS fallback
HF_TOKEN=                     # optional: a free HuggingFace token raises model-download rate limits
SRAG_API_TOKEN=               # set this before exposing the server beyond localhost
```

See `.env.example` for the full list.

---

## Embedding providers

The vector dimension belongs to the embedding model, so SRag namespaces its
tables by provider + model + dimension — `web_chunks__local__all-minilm-l6-v2__384`.
Switching provider creates a sibling table instead of corrupting the existing
index, and `index_meta.json` records which model built each one.

| Provider | Install | Needs | Text leaves the machine |
|---|---|---|---|
| `local` (default) | `pip install srag[local]` | ~800 MB of torch | No |
| `google` | nothing extra | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Yes, to Google |

```python
from srag import SRag, SRagConfig

# Local and offline — the default
sr = SRag(config=SRagConfig(embedding_provider="local"))

# Hosted embeddings: no torch, no local model, no CPU cost between calls
sr = SRag(config=SRagConfig(
    embedding_provider="google",
    embedding_model="text-embedding-004",
))
```

**Lean installs.** `pip install srag` no longer pulls torch. Local embeddings and
the CrossEncoder reranker need `srag[local]`; when it is missing SRag falls back
to budget-only chunking and skips reranking rather than failing, so a
Google-embedding deployment installs no torch at all. `import srag` resolves
heavy modules lazily — `SRagOrchestrator`, `SRagIndexer`, `DocumentIngestor` and
friends load on first use.

---

## Quickstart

```python
from srag import SRag

sr = SRag()

# Search and index
result = await sr.search("West Bengal election results 2026", session="wb_elections")

# Query the indexed session
chunks = await sr.query("who won the most seats?", session="wb_elections", k=5)

# Build structured context for an LLM
context = await sr.build_context("who won the most seats?", session="wb_elections")
print(context.to_prompt())

# Export for downstream systems
print(result.to_json())        # clean JSON
print(result.to_mongodb())     # MongoDB-ready document
print(result.to_jsonl())       # JSONL for streaming pipelines
print(result.trace.summary())  # fetch=210ms chunk=45ms embed=380ms rerank=90ms
```

Every retrieval call is a coroutine — `search()`, `query()`, `query_session()` and
`build_context()` — so it drops straight into an async agent or an MCP tool. In a
plain script, use the synchronous aliases: `query_sync()`,
`query_session_sync()`, `build_context_sync()`.

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

## Use it from an agent — MCP and HTTP

`srag` ships a FastAPI server, and with `srag[mcp]` that same app is mounted as an
MCP server. Any MCP client — Anu, Claude Desktop, Cursor — gets search, query and
ingest as tools without writing an adapter.

```bash
pip install "srag[mcp]"
srag serve --port 8000
# or: uvicorn srag.server:app --host 0.0.0.0 --port 8000
```

| Endpoint | What it is |
|---|---|
| `http://localhost:8000/health` | Liveness, plus whether MCP is mounted |
| `http://localhost:8000/mcp` | MCP over streamable HTTP — **preferred**, the default for modern clients |
| `http://localhost:8000/mcp/sse` | MCP over SSE — for clients that only speak SSE |
| `/search`, `/query`, `/ingest` | Plain HTTP, for anything that isn't an MCP client |

Point a client at it. Streamable HTTP needs nothing beyond the URL:

```json
{
  "mcpServers": {
    "srag": { "url": "http://localhost:8000/mcp" }
  }
}
```

SSE clients have to say so — a bare `url` is read as streamable HTTP, so pointing
one at `/mcp/sse` without `type` connects with the wrong transport and fails:

```json
{
  "mcpServers": {
    "srag": { "url": "http://localhost:8000/mcp/sse", "type": "sse" }
  }
}
```

The MCP surface is the search, retrieval, ingestion and session endpoints — eleven
tools. `/health`, `/` and `/inspect` are excluded on purpose: they are for humans
and uptime checks, and every tool you expose costs the agent prompt budget on
every single turn.

### Security

The server binds **`127.0.0.1`** by default. It has no authentication unless you
set one, and it reads local files on request, so loopback-only is the safe
default — pass `--host 0.0.0.0` deliberately, and set a token when you do.

| Variable | Default | What it does |
|---|---|---|
| `SRAG_API_TOKEN` | unset | Requires `Authorization: Bearer <token>` on every request except `/health` |
| `SRAG_ALLOWED_ORIGINS` | local dev origins | CORS allowlist, comma separated. `*` restores the old wildcard — don't, unless the browser app is trusted |
| `SRAG_INGEST_ROOTS` | unset (any path) | `os.pathsep`-separated directories `/ingest` may read from |

With a token set, an MCP client passes it as a header:

```json
{
  "mcpServers": {
    "srag": {
      "url": "http://localhost:8000/mcp",
      "headers": { "Authorization": "Bearer YOUR_TOKEN" }
    }
  }
}
```

Why the allowlist matters: before 1.3.9 the server answered `allow_origins=["*"]`
with no auth, so **any page a user visited** could POST `/ingest` with a path like
`/home/user/notes.txt` and then POST `/query` to read it back. CORS does not
affect MCP or other server-to-server clients — they send no `Origin` header — so
locking it down costs them nothing.

### Container, and a public URL for an agent

A container is the easiest way to keep SRag always-on and reachable from a hosted
agent such as Anu. The compose file ships ready:

```bash
cp .env.example .env
# set SRAG_API_TOKEN to something long and random, and SRAG_INGEST_ROOTS to the
# directories you actually want /ingest to read
docker compose up -d --build
curl -s -H "Authorization: Bearer $SRAG_API_TOKEN" localhost:8000/sessions
```

The container listens on `127.0.0.1` only, keeps its index in the `srag_data`
volume (`/app/srag_db`), and is only reported healthy once the indexer and embedding model are
loaded (`/sessions`, not `/health`) — so an agent that connects the moment the
container is up does not pay the model load on its first search.

To reach it from a hosted agent, put a tunnel in front. OpenTunnel is a good fit
because TLS terminates on your machine, so the relay forwards ciphertext and
never sees your token:

```bash
opentunnel route add 8000
# → https://<random>.opentunnel.xyz
```

**A tunnel is reachability, not hosting.** The public URL forwards to a bridge
that runs on the machine you started it from, and TLS terminates there — so the
machine has to stay on. Switch it off and the URL is dead. For a search backend
that is available the moment an agent asks, run this container on a host that
stays up: a VPS with the `srag_data` volume, or any platform that gives you a
persistent disk. A laptop is not one.

**The URL is not authentication.** Tunnel hostnames are public and published to
certificate transparency logs, so anyone who learns it can reach your server.
With `SRAG_API_TOKEN` set, everything except `/health` needs the bearer header;
`SRAG_INGEST_ROOTS` is what stops a stranger using `/ingest` to read your files.

Point Anu at the streamable HTTP endpoint:

```json
{
  "mcpServers": {
    "srag": {
      "url": "https://<random>.opentunnel.xyz/mcp",
      "headers": { "Authorization": "Bearer YOUR_SRAG_API_TOKEN" }
    }
  }
}
```

Two operational notes. A tunnel dies with your machine, so "always available"
really means "the host stays up" — a laptop that sleeps is not a search backend.
And if the tunnel drops and reconnects under a new hostname, update the connector:
the MCP client keeps its connection until the page reloads.

Two things worth knowing before you wire it up. **Run SRag as its own service, not
inside the agent's process** — it holds a LanceDB index and an embedding model, and
a chat app should not carry either. And **the free hosting tier that sleeps after an
hour will break tool calls**; the first call of the day hits a cold container and
fails mid-answer. Give this service somewhere that stays up.

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
│                          SearchLayer                            │
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

> `ingest()` is synchronous (unlike `search()` / `query()`), so call it without `await`.

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
    chunks = sr.query_sync("question", session="nonexistent")   # or: await sr.query(...)
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
| v1.1.0 | ✅ shipped | FastAPI HTTP server, Docker support, MCP server via `fastapi-mcp` |
| v1.2.0 | ✅ shipped | `DOMDistiller` structural HTML pruning, `DomainFilterStore` per-domain selector overrides, Playwright live-DOM path, `/inspect` and `/read` endpoints |
| v1.2.1 | ✅ shipped | Scraper `sympy` shadowing, `srag read` fallback, cross-document chunk dedup |
| v1.3.0 | ✅ shipped | Pluggable embeddings (local or Google), indexes namespaced by model, async retrieval API, torch-free base install |
| v1.3.1 | ✅ shipped | Documentation only — PyPI freezes the README at upload, so the 1.3.0 page still showed pre-MCP docs |
| v1.3.2 | ✅ shipped | `ingest()` crashed when a document produced no chunks |
| v1.3.3 | ✅ shipped | Coherence filter silently dropped your own short documents |
| v1.3.4 | ✅ shipped | `search()` returns `SRagResult` instead of a raw dict |
| v1.3.5 | ✅ shipped | Reputation update logged a failure on every search |
| v1.3.6 | ✅ shipped | `srag query` awaited — no more coroutine printed |
| v1.3.7 | ✅ shipped | `srag search` / `srag index` RecursionError, plus the CLI smoke tests that catch it |
| v1.3.8 | ✅ shipped | Dates no longer fabricated, `--check-conflicts` |
| v1.3.9 | ✅ shipped | Server hardening: loopback bind by default, bearer-token auth, CORS allowlist, `SRAG_INGEST_ROOTS`; fixed server import and MCP mount (see [CHANGELOG](https://github.com/square-box-hash/SRag/blob/main/CHANGELOG.md)) |
| v1.3.10 | ✅ shipped | Container serves MCP, keeps its index across restarts, and is only healthy once warm |

---

## Requirements

- Python 3.10+
- No API keys, no account, no hosted service

**The base install is deliberately lean.** `pip install srag` does not pull torch:
`import srag` resolves in ~150 ms and loads nothing heavy until a feature needs it.

| Extra | Adds | You need it for |
|---|---|---|
| *(none)* | core pipeline, LanceDB, SearXNG/DDGS search | search, index and query with Google embeddings |
| `srag[local]` | sentence-transformers + torch (~800 MB) | offline embeddings and the CrossEncoder reranker |
| `srag[mcp]` | FastAPI + `fastapi-mcp` | running SRag as an MCP server for an agent |
| `srag[docs]` | pymupdf, python-docx | ingesting PDFs and Word documents |
| `srag[databases]` | psycopg2-binary | ingesting PostgreSQL |
| `srag[all]` | everything above | — |

Without `srag[local]`, chunking falls back to token-budget packing and reranking is
skipped, rather than failing — which is the point of keeping torch optional.

---

## License

MIT — see [LICENSE](https://github.com/square-box-hash/SRag/blob/main/LICENSE)

---

<div align="center">

*Built as the retrieval layer for [Anu](https://github.com/square-box-hash) — a multi-specialist adaptive AI routing architecture.*

*Built by a class 11 student who got tired of paying for retrieval APIs.*

</div>