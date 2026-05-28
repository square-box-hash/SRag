import logging
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from srag.orchestrator import SRagOrchestrator
from srag.scraper import AnuInfrastructureScraper
from srag.indexer import SRagIndexer
from srag.config import SRagConfig
from srag.ingestor import DocumentIngestor
from srag.result import SRagResult, SRagTrace
from srag.tracer import SRagTracer
from srag.recency import RecencyRanker
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

__version__ = "1.0.0"
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
        )
        return {
            "session":       session,
            "source":        source,
            "doc_count":     len(docs),
            "chunk_count":   len(chunks),
            "indexed_count": stats.get("total_indexed", 0),
            "success":       len(chunks) > 0,
        }

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def query(
        self,
        query:   str,
        session: str,
        k:       int  = 5,
        debug:   bool = False,
    ) -> list[dict]:
        return self._orchestrator.query(query, session, k=k, debug=debug)

    def query_session(
        self,
        query:   str,
        session: str,
        k:       int  = 5,
        debug:   bool = False,
    ) -> list[dict]:
        return self._orchestrator.query_session(query, session, k=k, debug=debug)

    def build_context(
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