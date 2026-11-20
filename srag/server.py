"""
srag/server.py — SRag FastAPI Server (v1.3.10)

Wraps the SRag library as a language-agnostic HTTP service.
Any language that can make HTTP requests can now use SRag.

Run:
    uvicorn srag.server:app --host 127.0.0.1 --port 8000
    # or via CLI:
    srag serve --port 8000

Binding note: this service has no authentication by default, so it binds
127.0.0.1 (loopback only). Pass --host 0.0.0.0 deliberately to expose it on
your network, and set SRAG_API_TOKEN when you do.
"""

from __future__ import annotations

import hmac
import logging
import os
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("srag.server")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SRag API",
    description=(
        "Smart RAG — local retrieval as an HTTP service. "
        "Replaces Tavily, Firecrawl, and Perplexity with a single open-source pipeline."
    ),
    version="1.3.10",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Security: CORS allowlist and optional bearer token
# ---------------------------------------------------------------------------
#
# Why not allow_origins=["*"]: this server ingests local files and fetches URLs.
# With a wildcard, any web page a user visits could POST /ingest with a path
# like /home/user/notes.txt and then POST /query to read the contents back —
# local file disclosure driven from a third-party origin, no auth required.
#
# CORS is irrelevant to MCP and other server-to-server clients (they send no
# Origin header), so locking this down costs those callers nothing.


def _cors_origins() -> list[str]:
    """Explicit origins from SRAG_ALLOWED_ORIGINS, else local dev defaults."""
    raw = os.environ.get("SRAG_ALLOWED_ORIGINS", "").strip()
    if raw == "*":  # deliberate, documented opt-in
        return ["*"]
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


_API_TOKEN = os.environ.get("SRAG_API_TOKEN", "").strip()
# /health stays open so liveness probes keep working with auth enabled.
_OPEN_PATHS = {"/health"}


@app.middleware("http")
async def _require_bearer(request: Request, call_next):
    """Enforce SRAG_API_TOKEN when it is set. No-op when it is unset."""
    if not _API_TOKEN or request.url.path in _OPEN_PATHS:
        return await call_next(request)

    scheme, _, value = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(
        value.strip().encode(), _API_TOKEN.encode()
    ):
        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized. Send 'Authorization: Bearer <SRAG_API_TOKEN>'."},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)


# Added last so it is the outermost middleware — a 401 from the check above
# still carries CORS headers, and the browser shows the real error.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# ---------------------------------------------------------------------------
# Lazy SRag instance — initialised once on first request
# ---------------------------------------------------------------------------

_srag_instance = None


def _fail(context: str, exc: Exception, status: int = 500) -> HTTPException:
    """Log the real error, return a generic one.

    Exception text can carry filesystem paths, connection strings with
    credentials, and library internals. None of that belongs in an HTTP
    response; it belongs in the server log.
    """
    logger.exception("%s failed", context)
    return HTTPException(
        status_code=status,
        detail=f"{context} failed. See server logs for details.",
    )


def _ingest_path_allowed(path: str) -> bool:
    """True when `path` is inside one of SRAG_INGEST_ROOTS (or no roots set)."""
    raw = os.environ.get("SRAG_INGEST_ROOTS", "").strip()
    if not raw:
        return True
    try:
        target = os.path.realpath(path)
    except (OSError, ValueError):
        return False
    for root in raw.split(os.pathsep):
        root = root.strip()
        if not root:
            continue
        real_root = os.path.realpath(root)
        if target == real_root or target.startswith(real_root + os.sep):
            return True
    return False


def get_srag():
    global _srag_instance
    if _srag_instance is None:
        from srag import SRag  # type: ignore
        _srag_instance = SRag()
    return _srag_instance


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., description="The search query")
    session: str = Field(..., description="Session name — isolates this topic in LanceDB")
    results: int = Field(10, ge=1, le=50, description="Max results to fetch")
    force_new: bool = Field(False, description="Wipe and rebuild the session before searching")
    config_preset: str = Field("full", pattern="^(full|fast|lightweight)$",
                               description="SRagConfig preset to use")


class ParallelSearchRequest(BaseModel):
    queries: list[dict[str, Any]] = Field(
        ...,
        description='List of {"query": str, "session": str} dicts',
        json_schema_extra={
            "example": [
                {"query": "FastAPI tutorial", "session": "fastapi"},
                {"query": "LanceDB tutorial", "session": "lancedb"},
            ]
        },
    )
    config_preset: str = Field("full", pattern="^(full|fast|lightweight)$")


class SequentialSearchRequest(BaseModel):
    steps: list[dict[str, Any]] = Field(
        ...,
        description='Chained queries — use "depends_on" to inject context from a prior session',
        json_schema_extra={
            "example": [
                {"query": "GST rate India electronics", "session": "gst_base"},
                {"query": "GST filing deadline", "session": "gst_deadline", "depends_on": "gst_base"},
            ]
        },
    )
    config_preset: str = Field("full", pattern="^(full|fast|lightweight)$")


class VerifyRequest(BaseModel):
    query: str
    session: str
    config_preset: str = Field("full", pattern="^(full|fast|lightweight)$")


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question to answer from the session")
    session: str = Field(..., description="Session to query")
    k: int = Field(5, ge=1, le=50, description="Number of chunks to retrieve")


class ContextRequest(BaseModel):
    question: str
    session: str
    output_format: str = Field(
        "llm",
        pattern="^(llm|json|jsonl|mongodb)$",
        description=(
            "Output format:\n"
            "  llm      — clean structured context string, ready to paste into an LLM prompt\n"
            "  json     — typed SRagResult as JSON\n"
            "  jsonl    — JSONL for streaming pipelines\n"
            "  mongodb  — MongoDB-ready document"
        ),
    )


class IngestRequest(BaseModel):
    source: str = Field(
        ...,
        description=(
            "File path, folder path, or database URI.\n"
            "  File:     'report.pdf', 'notes.txt', 'data.csv'\n"
            "  Folder:   './docs/'\n"
            "  SQLite:   'sqlite:///mydb.sqlite'\n"
            "  Postgres: 'postgresql://user:pass@localhost/mydb'"
        ),
    )
    session: str
    table: str | None = Field(None, description="Database table (SQLite/Postgres)")
    query: str | None = Field(None, description="SQL query override (SQLite/Postgres)")


class StaleCheckResponse(BaseModel):
    session: str
    is_stale: bool
    max_age_hours: float


# ---------------------------------------------------------------------------
# LLM-native output formatter
# ---------------------------------------------------------------------------

def _format_llm_context(result: Any, question: str) -> dict:
    """
    Formats SRagResult into a clean, LLM-consumable structure.
    Designed so an LLM agent can consume this directly with no preprocessing.
    """
    chunks = result.chunks if hasattr(result, "chunks") else []
    sources = result.sources if hasattr(result, "sources") else []
    trace = result.trace.summary() if hasattr(result, "trace") and result.trace else None

    formatted_chunks = []
    for i, chunk in enumerate(chunks, 1):
        formatted_chunks.append({
            "index": i,
            "source": chunk.get("source") or chunk.get("url", "unknown"),
            "confidence": round(chunk.get("score", chunk.get("coherence_score", 0.0)), 3),
            "text": chunk.get("text", chunk.get("content", "")),
        })

    return {
        "query": question,
        "session": result.session if hasattr(result, "session") else None,
        "topic": result.topic if hasattr(result, "topic") else None,
        "retrieved_chunks": len(formatted_chunks),
        "sources_count": len(sources),
        "timing": trace,
        "conflicts": getattr(result, "conflicts", None),
        "context": formatted_chunks,
        # Flat prompt string — paste directly into an LLM
        "prompt_ready": _build_prompt_string(question, formatted_chunks),
    }


def _build_prompt_string(question: str, chunks: list[dict]) -> str:
    """Builds a clean, structured context string ready for direct LLM injection."""
    if not chunks:
        return f"[NO CONTEXT AVAILABLE]\nQuestion: {question}"

    lines = [
        "[RETRIEVAL CONTEXT]",
        f"question: {question}",
        f"chunks: {len(chunks)}",
        "",
    ]
    for chunk in chunks:
        lines.append(f"[{chunk['index']}] {chunk['source']} — confidence {chunk['confidence']}")
        lines.append(chunk["text"].strip())
        lines.append("")

    lines.append("[END CONTEXT]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health():
    """Liveness check."""
    return {"status": "ok", "version": "1.3.10", "service": "srag"}


@app.get("/", tags=["System"])
async def root():
    return {
        "service": "SRag API",
        "version": "1.3.10",
        "docs": "/docs",
        "mcp": {
            "available": _mcp_available,
            "http_endpoint": "/mcp" if _mcp_available else None,
            "sse_endpoint": "/mcp/sse" if _mcp_available else None,
            "install": "pip install fastapi-mcp" if not _mcp_available else None,
        },
        "endpoints": {
            "read":              "POST /read",
            "inspect":           "GET  /inspect",
            "search":            "POST /search",
            "parallel_search":   "POST /search/parallel",
            "sequential_search": "POST /search/sequential",
            "verify":            "POST /search/verify",
            "query":             "POST /query",
            "context":           "POST /context",
            "ingest":            "POST /ingest",
            "sessions":          "GET  /sessions",
            "session_stale":     "GET  /sessions/{name}/stale",
            "health":            "GET  /health",
        },
    }


# ---------------------------------------------------------------------------
# Search endpoints
# ---------------------------------------------------------------------------

@app.post("/search", tags=["Search"])
async def search(req: SearchRequest):
    """
    Run a single web search and index results into a session.
    Returns LLM-ready context + full metadata.
    """
    sr = get_srag()
    try:
        _apply_config(sr, req.config_preset)
        t0 = time.perf_counter()
        result = await sr.search(req.query, session=req.session, force_new=req.force_new)
        elapsed = round((time.perf_counter() - t0) * 1000)

        return {
            "success": result.success,
            "session": req.session,
            "query": req.query,
            "doc_count": getattr(result, "doc_count", None),
            "chunk_count": getattr(result, "chunk_count", None),
            "topic": getattr(result, "topic", None),
            "timing_ms": elapsed,
            "trace": result.trace.summary() if hasattr(result, "trace") and result.trace else None,
            "sources": getattr(result, "sources", []),
            "llm_context": _format_llm_context(result, req.query),
        }
    except Exception as e:
        raise _fail("/search", e)


@app.post("/search/parallel", tags=["Search"])
async def parallel_search(req: ParallelSearchRequest):
    """
    Run multiple independent queries concurrently.
    Each query gets its own session. Results returned in order.
    """
    sr = get_srag()
    try:
        _apply_config(sr, req.config_preset)
        results = await sr.parallel_search(req.queries)
        return {
            "success": True,
            "count": len(results),
            "results": [
                {
                    "query": q.get("query"),
                    "session": q.get("session"),
                    "success": r.success,
                    "chunk_count": getattr(r, "chunk_count", None),
                    "llm_context": _format_llm_context(r, q.get("query", "")),
                }
                for q, r in zip(req.queries, results)
            ],
        }
    except Exception as e:
        raise _fail("/search/parallel", e)


@app.post("/search/sequential", tags=["Search"])
async def sequential_search(req: SequentialSearchRequest):
    """
    Run chained queries where each step can depend on the prior step's results.
    Use 'depends_on' in a step to inject context from a previous session.
    """
    sr = get_srag()
    try:
        _apply_config(sr, req.config_preset)
        results = await sr.sequential_search(req.steps)
        return {
            "success": True,
            "steps": len(req.steps),
            "results": [
                {
                    "query": step.get("query"),
                    "session": step.get("session"),
                    "depends_on": step.get("depends_on"),
                    "success": r.success,
                    "chunk_count": getattr(r, "chunk_count", None),
                    "llm_context": _format_llm_context(r, step.get("query", "")),
                }
                for step, r in zip(req.steps, results)
            ],
        }
    except Exception as e:
        raise _fail("/search/sequential", e)


@app.post("/search/verify", tags=["Search"])
async def verify_search(req: VerifyRequest):
    """
    Conflict detection search — if two sources disagree, both are returned
    with structured conflict metadata. No silent resolution.
    """
    sr = get_srag()
    try:
        _apply_config(sr, req.config_preset)
        result = await sr.verify(req.query, session=req.session)
        return {
            "success": result.success,
            "session": req.session,
            "query": req.query,
            "status": getattr(result, "status", None),
            "conflict_detected": getattr(result, "status", "") == "conflict_detected",
            "conflicts": getattr(result, "conflicts", None),
            "llm_context": _format_llm_context(result, req.query),
        }
    except Exception as e:
        raise _fail("/search/verify", e)


# ---------------------------------------------------------------------------
# Query / Context endpoints
# ---------------------------------------------------------------------------

@app.post("/query", tags=["Retrieval"])
async def query_session(req: QueryRequest):
    """
    Semantic search inside an already-indexed session.
    Returns top-k chunks ranked by relevance.
    Does NOT trigger a new web search.
    """
    sr = get_srag()
    try:
        chunks = sr.query(req.question, session=req.session, k=req.k)
        formatted = [
            {
                "index": i + 1,
                "source": c.get("source") or c.get("url", "unknown"),
                "confidence": round(c.get("score", c.get("coherence_score", 0.0)), 3),
                "text": c.get("text", c.get("content", "")),
            }
            for i, c in enumerate(chunks)
        ]
        return {
            "session": req.session,
            "question": req.question,
            "retrieved": len(formatted),
            "chunks": formatted,
            "prompt_ready": _build_prompt_string(req.question, formatted),
        }
    except Exception as e:
        raise _fail("/query", e)


@app.post("/context", tags=["Retrieval"])
async def build_context(req: ContextRequest):
    """
    Build structured LLM context from an indexed session.

    output_format options:
    - llm      → clean structured string, paste directly into LLM prompt
    - json     → typed SRagResult as JSON
    - jsonl    → JSONL for streaming pipelines
    - mongodb  → MongoDB-ready document
    """
    sr = get_srag()
    try:
        context = sr.build_context(req.question, session=req.session)

        if req.output_format == "llm":
            return {"format": "llm", "output": context.to_prompt()}
        elif req.output_format == "json":
            return {"format": "json", "output": context.to_json()}
        elif req.output_format == "jsonl":
            return {"format": "jsonl", "output": context.to_jsonl()}
        elif req.output_format == "mongodb":
            return {"format": "mongodb", "output": context.to_mongodb()}
    except Exception as e:
        raise _fail("/context", e)


# ---------------------------------------------------------------------------
# Ingest endpoint
# ---------------------------------------------------------------------------

@app.post("/ingest", tags=["Ingestion"])
async def ingest(req: IngestRequest):
    """
    Ingest a local file, folder, or database into a session.

    Supported sources:
    - File path:    'report.pdf', 'notes.txt', 'data.csv', 'config.json', 'doc.docx'
    - Folder path:  './docs/'  (ingests all supported files recursively)
    - SQLite:       'sqlite:///mydb.sqlite'
    - PostgreSQL:   'postgresql://user:pass@localhost/mydb'

    Optional: pass 'table' or 'query' for database sources.
    """
    sr = get_srag()

    # Optional confinement: with SRAG_INGEST_ROOTS set (os.pathsep separated),
    # the endpoint refuses to read files outside those directories. Unset means
    # unchanged behaviour, so existing users are not broken by the fix.
    is_db_source = req.source.startswith(("sqlite://", "postgresql://", "postgres://"))
    if not is_db_source and not _ingest_path_allowed(req.source):
        raise HTTPException(
            status_code=403,
            detail=(
                "Path is outside SRAG_INGEST_ROOTS. "
                "Set SRAG_INGEST_ROOTS to the directories ingestion may read."
            ),
        )

    try:
        kwargs = {}
        if req.table:
            kwargs["table"] = req.table
        if req.query:
            kwargs["query"] = req.query

        t0 = time.perf_counter()
        sr.ingest(req.source, session=req.session, **kwargs)
        elapsed = round((time.perf_counter() - t0) * 1000)

        return {
            "success": True,
            "source": req.source,
            "session": req.session,
            "timing_ms": elapsed,
        }
    except Exception as e:
        raise _fail("/ingest", e)


# ---------------------------------------------------------------------------
# Session management endpoints
# ---------------------------------------------------------------------------

@app.get("/sessions", tags=["Sessions"])
async def list_sessions():
    """List all indexed sessions."""
    sr = get_srag()
    try:
        sessions = sr.list_sessions()
        return {"sessions": sessions, "count": len(sessions)}
    except Exception as e:
        raise _fail("/sessions", e)


@app.get("/sessions/{name}/stale", tags=["Sessions"], response_model=StaleCheckResponse)
async def check_stale(
    name: str,
    max_age_hours: float = Query(24.0, ge=0.1, description="TTL in hours"),
):
    """Check if a session is older than the given TTL."""
    sr = get_srag()
    try:
        stale = sr.is_stale(name, max_age_hours=max_age_hours)
        return StaleCheckResponse(session=name, is_stale=stale, max_age_hours=max_age_hours)
    except Exception as e:
        raise _fail("/sessions/{name}/stale", e)


@app.delete("/sessions/{name}", tags=["Sessions"])
async def delete_session(name: str):
    """
    Delete a session's index and its metadata entry.

    Answers 404 when no session of that name is indexed. The next search with
    the same session name rebuilds it from scratch.
    """
    sr = get_srag()
    try:
        deleted = sr.delete_session(name)
    except AttributeError:
        raise HTTPException(
            status_code=501,
            detail="This SRag build has no delete_session(); upgrade to >= 1.3.9.",
        )
    except Exception as e:
        raise _fail("/sessions/{name}", e)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"No indexed session named '{name}'.")
    return {"session": name, "deleted": True}


# ---------------------------------------------------------------------------
# Inspect endpoint
# ---------------------------------------------------------------------------

@app.get("/inspect", tags=["System"])
async def inspect(
    show:    str           = Query("all", description="What to show: all | reputation | lexicon | candidates | chunks | domain-filters"),
    session: str | None    = Query(None,  description="Session name — required for show=chunks"),
    domain:  str | None    = Query(None,  description="Filter reputation by domain substring"),
    topic:   str | None    = Query(None,  description="Filter reputation/lexicon by topic"),
):
    """
    System introspection — exposes the same data as `srag inspect` over HTTP.
    Returns reputation scores, lexicon state, chunk stats, and domain filter flags as JSON.
    """
    sr = get_srag()
    try:
        from srag.inspector import inspect as srag_inspect
        db   = sr._orchestrator.indexer.db
        data = srag_inspect(db=db, show=show, session=session, domain=domain, topic=topic)
        return data
    except Exception as e:
        raise _fail("/inspect", e)


# ---------------------------------------------------------------------------
# Read endpoint
# ---------------------------------------------------------------------------

class ReadRequest(BaseModel):
    url: str = Field(..., description="URL to fetch, distill, and return as clean text")


@app.post("/read", tags=["Retrieval"])
async def read_url(req: ReadRequest):
    """
    Fetch a URL and return distilled clean text — without indexing it.
    HTTP equivalent of `srag read <url>`.
    """
    sr = get_srag()
    try:
        import httpx
        from srag.dom_distiller import DOMDistiller
        from srag.scraper import REALISTIC_HEADERS
        import random

        domain_filters = getattr(sr._orchestrator, "domain_filters", None)
        distiller      = DOMDistiller(domain_filters=domain_filters)

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(req.url, headers=random.choice(REALISTIC_HEADERS))
            resp.raise_for_status()
            html = resp.text

        distilled = distiller.distill(html, req.url)

        import trafilatura
        text = trafilatura.extract(
            distilled["html"],
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_recall=True,
        ) or ""

        from bs4 import BeautifulSoup
        soup  = BeautifulSoup(distilled["html"], "html.parser")
        title = (soup.title.string.strip() if soup.title and soup.title.string else req.url)

        return {
            "url":            req.url,
            "title":          title,
            "content":        text,
            "content_length": len(text),
            "noise_removed":  distilled["removed"],
        }
    except httpx.HTTPStatusError as e:
        raise _fail("/read", e, status=e.response.status_code)
    except Exception as e:
        raise _fail("/read", e)

# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def _apply_config(sr: Any, preset: str) -> None:
    """Apply a SRagConfig preset to the running instance."""
    try:
        from srag import SRagConfig  # type: ignore
        preset_map = {
            "full": SRagConfig.full,
            "fast": SRagConfig.fast,
            "lightweight": SRagConfig.lightweight,
        }
        if preset in preset_map:
            sr.config = preset_map[preset]()
    except Exception:
        pass  # if config can't be applied, run with existing config


# ---------------------------------------------------------------------------
# CLI entrypoint (used by `srag serve`)
# ---------------------------------------------------------------------------

def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Start the SRag HTTP server. Called by `srag serve` CLI command."""
    uvicorn.run(
        "srag.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    serve()


# ---------------------------------------------------------------------------
# MCP Server — mounts at /mcp (streamable HTTP) and /mcp/sse (SSE)
# Exposes the search, retrieval, ingestion and session endpoints as MCP tools.
# Connect any MCP client (Anu, Claude Desktop, Cursor) to:
#   http://localhost:8000/mcp        <- streamable HTTP, preferred
#   http://localhost:8000/mcp/sse    <- SSE only clients
#
# Anu / modern clients:
#   { "mcpServers": { "srag": { "url": "http://localhost:8000/mcp" } } }
# SSE clients must declare the transport:
#   { "mcpServers": { "srag": { "url": "http://localhost:8000/mcp/sse", "type": "sse" } } }
# ---------------------------------------------------------------------------

def _mount_mcp() -> bool:
    """Mount the MCP server on both transports.

    /mcp      — streamable HTTP (what modern clients, including Anu, default to)
    /mcp/sse  — SSE (kept for Claude Desktop / Cursor configs already in the wild)

    Never raises: a broken or older fastapi-mcp must not stop the HTTP API from
    booting. Only a missing package is quiet; anything else is logged.
    """
    try:
        from fastapi_mcp import FastApiMCP
    except ImportError:
        return False

    try:
        mcp = FastApiMCP(
            app,
            name="SRag",
            description=(
                "Smart RAG — local retrieval pipeline. "
                "Search the web, query indexed sessions, ingest files and databases. "
                "No API keys required, runs fully local."
            ),
            # /health, / and /inspect are for humans and uptime checks. Exposing
            # them as tools spends the agent's prompt budget on every turn.
            exclude_tags=["System"],
        )
        mounted = False
        if hasattr(mcp, "mount_http"):
            mcp.mount_http(mount_path="/mcp")
            mounted = True
        if hasattr(mcp, "mount_sse"):
            mcp.mount_sse(mount_path="/mcp/sse")
            mounted = True
        if not mounted:
            # fastapi-mcp < 0.4 only has the deprecated SSE-only mount()
            mcp.mount(mount_path="/mcp")
            mounted = True
        return mounted
    except Exception as exc:  # noqa: BLE001 — must not take the API down
        import logging

        logging.getLogger(__name__).warning(
            "MCP mount failed (%s: %s); HTTP API is still available.",
            type(exc).__name__,
            exc,
        )
        return False

_mcp_available = _mount_mcp()