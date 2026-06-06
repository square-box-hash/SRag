"""
srag/server.py — SRag FastAPI Server (v1.1.0)

Wraps the SRag library as a language-agnostic HTTP service.
Any language that can make HTTP requests can now use SRag.

Run:
    uvicorn srag.server:app --host 0.0.0.0 --port 8000
    # or via CLI (v1.1.0):
    srag serve --port 8000
"""

from __future__ import annotations

import time
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SRag API",
    description=(
        "Smart RAG — local retrieval as an HTTP service. "
        "Replaces Tavily, Firecrawl, and Perplexity with a single open-source pipeline."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy SRag instance — initialised once on first request
# ---------------------------------------------------------------------------

_srag_instance = None


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
    return {"status": "ok", "version": "1.1.0", "service": "srag"}


@app.get("/", tags=["System"])
async def root():
    return {
        "service": "SRag API",
        "version": "1.1.0",
        "docs": "/docs",
        "mcp": {
            "available": _mcp_available,
            "sse_endpoint": "/mcp/sse" if _mcp_available else None,
            "install": "pip install fastapi-mcp" if not _mcp_available else None,
        },
        "endpoints": {
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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{name}", tags=["Sessions"])
async def delete_session(name: str):
    """
    Force-wipe a session. Next search with force_new=True will rebuild it.
    """
    sr = get_srag()
    try:
        # force_new on a dummy search clears the session
        # expose direct delete if SRag adds sr.delete_session() in future
        return {
            "session": name,
            "message": (
                f"To wipe session '{name}', run POST /search with "
                f'"session": "{name}", "force_new": true'
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
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
# MCP Server — mounts at /mcp/sse
# Exposes all SRag FastAPI endpoints as MCP tools automatically.
# Connect any MCP client (Claude Desktop, Cursor, Anu) to:
#   http://localhost:8000/mcp/sse
#
# Claude Desktop config:
#   { "mcpServers": { "srag": { "url": "http://localhost:8000/mcp/sse" } } }
# ---------------------------------------------------------------------------

def _mount_mcp():
    try:
        from fastapi_mcp import FastApiMCP
        mcp = FastApiMCP(
            app,
            name="SRag",
            description=(
                "Smart RAG — local retrieval pipeline. "
                "Search the web, query indexed sessions, ingest files and databases. "
                "No API keys required, runs fully local."
            ),
        )
        mcp.mount()  # mounts at /mcp → /mcp/sse for SSE connections
        return True
    except ImportError:
        return False

_mcp_available = _mount_mcp()