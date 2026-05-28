# srag/result.py
"""
SRagResult — typed return object for all SRag operations.
Replaces raw dicts with a consistent, IDE-friendly interface.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SRagTrace:
    """Per-step timing and debug info."""
    fetch_ms:    float = 0.0
    chunk_ms:    float = 0.0
    embed_ms:    float = 0.0
    rerank_ms:   float = 0.0
    total_ms:    float = 0.0
    doc_count:   int   = 0
    chunk_count: int   = 0
    topic:       str   = ""
    mode:        str   = "simple"  # simple | parallel | sequential | verify | ingest

    def summary(self) -> str:
        return (
            f"total={self.total_ms:.0f}ms "
            f"fetch={self.fetch_ms:.0f}ms "
            f"chunk={self.chunk_ms:.0f}ms "
            f"embed={self.embed_ms:.0f}ms "
            f"rerank={self.rerank_ms:.0f}ms "
            f"docs={self.doc_count} chunks={self.chunk_count}"
        )


@dataclass
class SRagResult:
    """
    Typed return object for all SRag search, ingest, and query operations.
    
    Usage:
        result = await sr.search("query", session="s")
        if result.success:
            print(result.context.to_prompt())
            print(result.trace.summary())
    """
    # ── Core fields ───────────────────────────────────────────────────────────
    success:       bool             = False
    session:       str              = ""
    query:         str              = ""
    mode:          str              = "simple"

    # ── Data ──────────────────────────────────────────────────────────────────
    docs:          list[dict]       = field(default_factory=list)
    chunks:        list[dict]       = field(default_factory=list)
    sources:       list[dict]       = field(default_factory=list)
    context:       Any              = None   # BuiltContext from context_builder

    # ── Stats ─────────────────────────────────────────────────────────────────
    doc_count:     int              = 0
    chunk_count:   int              = 0
    indexed_count: int              = 0
    topic:         str              = ""

    # ── Trace ─────────────────────────────────────────────────────────────────
    trace:         SRagTrace        = field(default_factory=SRagTrace)

    # ── Error info ────────────────────────────────────────────────────────────
    error:         Optional[str]    = None
    reason:        Optional[str]    = None

    # ── Conflict detection (verify mode) ─────────────────────────────────────
    status:        Optional[str]    = None
    conflicts:     list[dict]       = field(default_factory=list)

    def to_dict(self) -> dict:
        """Structured dict — ready for MongoDB, Redis, or any KV store."""
        return {
            "success":       self.success,
            "session":       self.session,
            "query":         self.query,
            "mode":          self.mode,
            "doc_count":     self.doc_count,
            "chunk_count":   self.chunk_count,
            "indexed_count": self.indexed_count,
            "topic":         self.topic,
            "sources":       self.sources,
            "chunks":        self.chunks,
            "error":         self.error,
            "reason":        self.reason,
            "status":        self.status,
            "conflicts":     self.conflicts,
            "trace":         {
                "total_ms":  self.trace.total_ms,
                "fetch_ms":  self.trace.fetch_ms,
                "chunk_ms":  self.trace.chunk_ms,
                "embed_ms":  self.trace.embed_ms,
                "rerank_ms": self.trace.rerank_ms,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Clean JSON string — ready for any downstream consumer."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_jsonl(self) -> str:
        """One JSON line per chunk — ready for JSONL pipelines."""
        lines = []
        for chunk in self.chunks:
            lines.append(json.dumps({
                "session": self.session,
                "query":   self.query,
                "topic":   self.topic,
                **chunk,
            }, default=str))
        return "\n".join(lines)

    def to_mongodb(self) -> dict:
        """
        MongoDB-ready document.
        Drop this directly into a collection.
        """
        doc = self.to_dict()
        doc["_id"] = f"{self.session}:{self.query[:40]}"
        return doc

    def to_prompt(self) -> str:
        """
        Ready-to-inject LLM prompt context.
        Falls back to building from chunks if BuiltContext unavailable.
        """
        if self.context and hasattr(self.context, "to_prompt"):
            return self.context.to_prompt()

        if not self.chunks:
            return ""

        parts = []
        for i, chunk in enumerate(self.chunks[:5]):
            parts.append(
                f"[{i+1}] {chunk.get('title', '')}\n"
                f"Source: {chunk.get('source', '')}\n"
                f"{chunk.get('content', '')[:600]}"
            )
        return "\n\n".join(parts)

    def __bool__(self) -> bool:
        return self.success

    def __repr__(self) -> str:
        return (
            f"SRagResult(success={self.success}, "
            f"session={self.session!r}, "
            f"docs={self.doc_count}, "
            f"chunks={self.chunk_count}, "
            f"topic={self.topic!r})"
        )