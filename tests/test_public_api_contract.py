"""Contract tests for the public API — the layer that keeps breaking.

These exist because three separate bugs shipped in one session that all had green
unit tests: `search()` returned a dict instead of `SRagResult`, `query()` became a
coroutine while the CLI still called it synchronously, and `ingest()` crashed on an
empty result. Every one of them was a public-API shape change that the internal
tests never touched.

No torch, no network, no LanceDB — the orchestrator is stubbed.
"""

import asyncio
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srag import SRag, SRagResult  # noqa: E402


class StubOrchestrator:
    """Mimics the orchestrator's real return shapes."""

    def __init__(self):
        self.calls = []

    async def search(self, query, session, force_new=False, max_results=None, debug=False):
        self.calls.append(("search", query, session))
        return {
            "session": session,
            "query": query,
            "topic": "general",
            "doc_count": 2,
            "chunk_count": 5,
            "indexed_count": 4,
            "success": True,
            "docs": [
                {"url": "https://a.example", "title": "A"},
                {"url": "https://b.example", "title": "B"},
                {"url": "https://a.example", "title": "A again"},
            ],
            "chunks": [{"content": "alpha"}, {"content": "beta"}],
        }

    async def verify(self, query, session, debug=False):
        return {"session": session, "query": query, "success": True, "docs": [], "chunks": []}

    async def parallel_search(self, plan, debug=False):
        return [{"session": p["session"], "query": p["query"], "success": True, "docs": [], "chunks": []} for p in plan]

    async def sequential_search(self, steps, debug=False):
        return [{"session": s["session"], "query": s["query"], "success": True, "docs": [], "chunks": []} for s in steps]


def make_srag():
    """Build an SRag without loading models, by injecting the orchestrator."""
    sr = object.__new__(SRag)
    sr._orchestrator = StubOrchestrator()
    sr._ingestor = None
    return sr


def test_search_returns_sragresult_not_dict():
    sr = make_srag()
    result = asyncio.run(sr.search("anything", session="s1"))

    assert isinstance(result, SRagResult), f"search() returned {type(result).__name__}, not SRagResult"
    assert result.success is True
    assert result.doc_count == 2 and result.chunk_count == 5
    assert result.raw, "raw payload should be preserved for --debug output"
    assert result.to_json()  # the documented serialiser must work


def test_search_derives_sources_and_dedupes():
    sr = make_srag()
    result = asyncio.run(sr.search("anything", session="s1"))

    urls = [s["url"] for s in result.sources]
    assert urls == ["https://a.example", "https://b.example"], urls


def test_verify_and_batch_modes_also_return_sragresult():
    sr = make_srag()

    assert isinstance(asyncio.run(sr.verify("q", session="s")), SRagResult)

    parallel = asyncio.run(sr.parallel_search([{"query": "a", "session": "s1"}]))
    assert all(isinstance(item, SRagResult) for item in parallel), parallel

    sequential = asyncio.run(sr.sequential_search([{"query": "a", "session": "s1"}]))
    assert all(isinstance(item, SRagResult) for item in sequential), sequential


def test_retrieval_is_async_and_sync_aliases_exist():
    for name in ("search", "query", "query_session", "build_context"):
        assert inspect.iscoroutinefunction(getattr(SRag, name)), f"{name}() should be a coroutine"

    for name in ("query_sync", "query_session_sync", "build_context_sync"):
        assert not inspect.iscoroutinefunction(getattr(SRag, name)), f"{name}() should be sync"


def test_ingest_survives_an_empty_indexer_result():
    """ingest() must not call .get() on None when nothing is indexed."""
    sr = object.__new__(SRag)

    class StubIngestor:
        def ingest(self, source, **kwargs):
            return [{"content": "a short note", "title": "note"}]

    class StubChunker:
        min_chunk_chars = 100

        def chunk_docs(self, docs):
            return []

    class StubIndexer:
        def index_documents(self, chunks, **kwargs):
            return None  # the old behaviour that caused the crash

    class StubOrch:
        def __init__(self):
            self.chunker = StubChunker()
            self.indexer = StubIndexer()
            self.config = type("C", (), {"ingest_min_coherence": 0.0})()

    sr._ingestor = StubIngestor()
    sr._orchestrator = StubOrch()

    stats = sr.ingest("note.txt", session="notes")
    assert stats["indexed_count"] == 0
    assert stats["success"] is False
    assert stats["doc_count"] == 1


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL {name}: {exc}")
    print("\nFAILURES:", failures if failures else "none")
    sys.exit(1 if failures else 0)
