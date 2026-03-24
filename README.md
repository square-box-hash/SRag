# SRag (Smart RAG) v0.5.0

SRag is a local, self-hosted Python library that provides web search, scraping, and vector indexing in a single pipeline — purpose-built as the retrieval layer for LLM/RAG workflows.

Built as a free alternative to Tavily, Firecrawl, and Perplexity Search APIs.

---

## Features

### Core Pipeline
- 🔎 **Web search** via DuckDuckGo — no API key needed, completely free
- ⚡ **Async scraping** via `httpx` with persistent connection pooling
- 🧹 **Smart content extraction** — trafilatura primary, BeautifulSoup fallback
- 🧠 **Vector indexing** via LanceDB + sentence-transformers
- 📁 **Session isolation** — each research topic gets its own table, never mixed

### Search Modes
- **Single search** — one query, one session
- **Parallel search** — multiple independent queries concurrently (configurable cap)
- **Sequential search** — chained queries where result of step A informs step B
- **Verification search** — conflict detection across sources, structured conflict JSON

### Reliability
- 🔁 Retry with exponential backoff on failed fetches
- 🚫 Blocked domain filtering — skips known 403 offenders automatically
- ⭐ Priority domain sorting — high quality sources fetched first
- 🔀 Redirect following — handles redirect chains automatically
- 🔇 Empty content filtering — blank pages never reach the vector store

### Cache Management
- `is_stale()` — check if a session is older than a configurable TTL
- `list_sessions()` — see all indexed sessions
- `force_new=True` — wipe and rebuild a specific session without touching others

### CLI
```bash
srag search "query"
srag index "query" --session name
srag query "question" --session name
srag verify "query" --session name
srag sessions
srag stale session_name
```

---

## Installation
```bash
# From GitHub
pip install git+https://github.com/square-box-hash/SRag.git@v0.5.0

# Local development
git clone https://github.com/square-box-hash/SRag.git
cd SRag
pip install -e .
```

---

## Setup

Copy `.env.example` to `.env` and add your HuggingFace token:
```bash
cp .env.example .env
```
```
HF_TOKEN=your_token_here
```

---

## Usage
```python
from srag import SRag

sr = SRag()

# Single search
result = await sr.search("python asyncio tutorial", session="python_async")

# Parallel search
results = await sr.parallel_search([
    {"query": "FastAPI tutorial", "session": "fastapi"},
    {"query": "LanceDB tutorial", "session": "lancedb"},
])

# Sequential search — step 2 informed by step 1
results = await sr.sequential_search([
    {"query": "GST rate India electronics", "session": "gst_base"},
    {"query": "GST filing deadline", "session": "gst_deadline", "depends_on": "gst_base"},
])

# Verification search — detects conflicting sources
result = await sr.verify("GST rate electronics", session="gst_verify")

# Query a stored session
chunks = sr.query("what is the penalty?", session="gst_base", k=5)

# Cache management
if sr.is_stale("gst_base", max_age_hours=24):
    await sr.search("GST rate India electronics", session="gst_base", force_new=True)

# List all sessions
print(sr.list_sessions())
```

---

## Architecture
```
Query
  ↓
SRag (orchestrator)
  ↓                    
DuckDuckGo (DDGS)      ← URL discovery
  ↓
httpx async            ← parallel fetching
  ↓
trafilatura / BS4      ← content extraction
  ↓
sentence-transformers  ← embedding (all-MiniLM-L6-v2, 384 dims)
  ↓
LanceDB                ← vector storage, session-isolated tables
  ↓
query_session()        ← semantic retrieval → JSON chunks
```

---

## Roadmap

| Version | Focus |
|---------|-------|
| ✅ v0.4.0 | Core pipeline — search, scrape, index, CLI |
| ✅ v0.5.0 | Parallel + sequential + verification search, unified API |
| 🔜 v0.6.0 | Playwright headless browser for JS-heavy pages, semantic chunking, query expansion |
| 🔜 v0.7.0 | Reranker, caching, source validation, domain-aware search |
| 🔜 v0.8.0 | Adaptive concurrency, smart context builder, token budgeting |
| 🔜 v0.9.0 | Trace logs, timing info, comprehensive error handling |
| 🔜 v1.0.0 | PyPI publish — stable, documented, production-ready |

---

## Requirements

- Python 3.10+
- Dependencies installed automatically via pip

---

## License

MIT License — see [LICENSE](LICENSE)

---

*Built as the retrieval layer for [Anu](https://github.com/square-box-hash) — a multi-specialist AI routing architecture.*