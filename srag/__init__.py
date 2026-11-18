from __future__ import annotations

import logging
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from srag.embeddings import (
    Embedder,
    EmbeddingMismatchError,
    EmbeddingSpec,
    GoogleEmbedder,
    LocalEmbedder,
    build_embedder,
)
from srag.config import SRagConfig
from srag.result import SRagResult, SRagTrace

# Everything that pulls in torch, playwright or the scraper stack is imported on
# first access, so `import srag` costs a fraction of what it used to.
_LAZY_EXPORTS = {
    "SRagOrchestrator": ("srag.orchestrator", "SRagOrchestrator"),
    "AnuInfrastructureScraper": ("srag.scraper", "AnuInfrastructureScraper"),
    "SRagIndexer": ("srag.indexer", "SRagIndexer"),
    "DocumentIngestor": ("srag.ingestor", "DocumentIngestor"),
    "SRagTracer": ("srag.tracer", "SRagTracer"),
    "RecencyRanker": ("srag.recency", "RecencyRanker"),
}


def __getattr__(name: str):
    """PEP 562 — resolve heavy exports only when something actually asks for them."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'srag' has no attribute {name!r}")
    from importlib import import_module

    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
from srag.exceptions import (
    SRagError,
    SRagFetchError,
    SRagTimeoutError,
    SRagBlockedError,
    SRagQualityError,
    SRagNoContentError,
    SRagIndexError,
    SRagSessionNotFoundError,
    SRagIngestError,
    SRagUnsupportedFormatError,
    SRagMissingDependencyError,
    SRagConfigError,
)

logger = logging.getLogger(__name__)

__version__ = "1.3.2"
__all__ = [
    "SRag",
    "SRagConfig",
    "SRagResult",
    "SRagTrace",
    "SRagTracer",
    "RecencyRanker",
    "DocumentIngestor",
    "SRagOrchestrator",
    "SRagError",
    "SRagFetchError",
    "SRagTimeoutError",
    "SRagBlockedError",
    "SRagQualityError",
    "SRagNoContentError",
    "SRagIndexError",
    "SRagSessionNotFoundError",
    "SRagIngestError",
    "SRagUnsupportedFormatError",
    "SRagMissingDependencyError",
    "SRagConfigError",
]


class SRag:
    """
    Unified public API for SRag.
    This is the only import Anu (or any user) needs.
    """

    def __init__(
        self,
        config:         SRagConfig = None,
        db_path:        str        = "./srag_db",
        max_results:    int        = 12,
        max_chars:      int        = 2000,
        extract_mode:   str        = "trafilatura",
        max_concurrent: int        = 5,
    ):
        # Imported here so the package itself stays importable without the
        # scraper stack, playwright or torch.
        from srag.ingestor import DocumentIngestor
        from srag.orchestrator import SRagOrchestrator

        self._orchestrator = SRagOrchestrator(
            config         = config,
            max_results    = max_results,
            max_chars      = max_chars,
            extract_mode   = extract_mode,
            max_concurrent = max_concurrent,
            db_path        = db_path,
        )
        self._ingestor = DocumentIngestor(config=self._orchestrator.config)

    # ── Search modes ──────────────────────────────────────────────────────────

    async def search(
        self,
        query:       str,
        session:     str,
        force_new:   bool          = False,
        max_results: Optional[int] = None,
        debug:       bool          = False,
    ) -> SRagResult:
        return await self._orchestrator.search(
            query       = query,
            session     = session,
            force_new   = force_new,
            max_results = max_results,
            debug       = debug,
        )

    async def parallel_search(self, plan: list, debug: bool = False) -> list:
        return await self._orchestrator.parallel_search(plan, debug=debug)

    async def sequential_search(self, steps: list, debug: bool = False) -> list:
        return await self._orchestrator.sequential_search(steps, debug=debug)

    async def verify(
        self,
        query:   str,
        session: str,
        debug:   bool = False,
    ) -> SRagResult:
        return await self._orchestrator.verify(query, session, debug=debug)

    # ── Local file + DB ingestion ─────────────────────────────────────────────

    def ingest(
        self,
        source:    str,
        session:   str,
        force_new: bool = False,
        **kwargs,
    ) -> dict:
        """
        Ingest a local file, folder, or database into a session.
        source: file path, folder path, sqlite:// or postgresql:// URI
        """
        docs   = self._ingestor.ingest(source, **kwargs)
        chunks = self._orchestrator.chunker.chunk_docs(docs)
        stats  = self._orchestrator.indexer.index_documents(
            chunks,
            table_name = session,
            force_new  = force_new,
        ) or {}

        indexed = int(stats.get("total_indexed", 0))
        if docs and not chunks:
            # The usual cause is a document shorter than min_chunk_chars.
            min_chars = getattr(self._orchestrator.chunker, "min_chunk_chars", 100)
            print(
                f"⚠️  Read {len(docs)} document(s) but produced no chunks. "
                f"Content may be shorter than min_chunk_chars={min_chars}, or contain "
                f"no sentence boundaries. Lower SRagConfig.min_chunk_chars to index it."
            )

        return {
            "session":       session,
            "source":        source,
            "doc_count":     len(docs),
            "chunk_count":   len(chunks),
            "indexed_count": indexed,
            "success":       indexed > 0,
        }

    # ── Retrieval ─────────────────────────────────────────────────────────────

    async def query(
        self,
        query:   str,
        session: str,
        k:       int  = 5,
        debug:   bool = False,
    ) -> list[dict]:
        """Async so every retrieval entry point matches search()."""
        return self._orchestrator.query(query, session, k=k, debug=debug)

    async def query_session(
        self,
        query:   str,
        session: str,
        k:       int  = 5,
        debug:   bool = False,
    ) -> list[dict]:
        return self._orchestrator.query_session(query, session, k=k, debug=debug)

    async def build_context(
        self,
        query:        str,
        session:      str,
        k:            int           = 10,
        token_budget: Optional[int] = None,
        debug:        bool          = False,
    ):
        return self._orchestrator.build_context(
            query, session, k=k, token_budget=token_budget, debug=debug
        )

    # ── Synchronous aliases (pre-1.3 callers, and scripts) ────────────────────

    def query_sync(self, query: str, session: str, k: int = 5, debug: bool = False) -> list[dict]:
        return self._orchestrator.query(query, session, k=k, debug=debug)

    def query_session_sync(
        self, query: str, session: str, k: int = 5, debug: bool = False
    ) -> list[dict]:
        return self._orchestrator.query_session(query, session, k=k, debug=debug)

    def build_context_sync(
        self,
        query:        str,
        session:      str,
        k:            int           = 10,
        token_budget: Optional[int] = None,
        debug:        bool          = False,
    ):
        return self._orchestrator.build_context(
            query, session, k=k, token_budget=token_budget, debug=debug
        )

    # ── Cache management ──────────────────────────────────────────────────────

    def is_stale(self, session: str, max_age_hours: int = 24) -> bool:
        return self._orchestrator.is_stale(session, max_age_hours)

    def list_sessions(self) -> list:
        return self._orchestrator.list_sessions()

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_scraper(self) -> AnuInfrastructureScraper:
        return self._orchestrator._make_scraper()

    def get_indexer(self) -> SRagIndexer:
        return self._orchestrator.indexer

    # ── Config access ─────────────────────────────────────────────────────────

    @property
    def config(self) -> SRagConfig:
        return self._orchestrator.config

    def __repr__(self) -> str:
        return (
            f"SRag(version={__version__!r}, "
            f"db={self._orchestrator.config.db_path!r}, "
            f"reranker={self._orchestrator.config.use_reranker}, "
            f"searxng={self._orchestrator.config.use_searxng})"
        )