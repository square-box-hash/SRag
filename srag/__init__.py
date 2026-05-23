from dotenv import load_dotenv
load_dotenv()

from srag.orchestrator import SRagOrchestrator
from srag.scraper import AnuInfrastructureScraper
from srag.indexer import SRagIndexer
from srag.config import SRagConfig
from srag.ingestor import DocumentIngestor

class SRag:
    """
    Unified public API for SRag.
    This is the only import Anu (or any user) needs.
    """
    def __init__(
        self,
        config: SRagConfig = None,
        db_path: str = "./srag_db",
        max_results: int = 12,
        max_chars: int = 2000,
        extract_mode: str = "trafilatura",
        max_concurrent: int = 5,
    ):
        self._orchestrator = SRagOrchestrator(
            config: SRagConfig = None,
            max_results=max_results,
            max_chars=max_chars,
            extract_mode=extract_mode,
            max_concurrent=max_concurrent,
            db_path=db_path,
        )
        self._ingestor = DocumentIngestor(config=self._orchestrator.config)

    # ── Search modes ──────────────────────────────────────────────────────────
    async def search(self, query, session, force_new=False, debug=False):
        return await self._orchestrator.search(query, session, force_new, debug)

    async def parallel_search(self, plan: list) -> list:
        return await self._orchestrator.parallel_search(plan)

    async def sequential_search(self, steps: list) -> list:
        return await self._orchestrator.sequential_search(steps)

    async def verify(self, query, session, debug=False):
        return await self._orchestrator.verify(query, session, debug=debug)

    # ── Local file + DB ingestion ─────────────────────────────────────────────
    def ingest(self, source: str, session: str, force_new: bool = False, **kwargs) -> dict:
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
    def get_context(self, session: str) -> dict:
        return self._orchestrator.get_context(session)
    
    def get_chunks(self, session: str) -> list:
        return self._orchestrator.get_chunks(session)

    def get_docs(self, session: str) -> list:
        return self._orchestrator.get_docs(session)
    
    def get_doc_count(self, session: str) -> int:
        return self._orchestrator.get_doc_count(session)
    
    def get_chunk_count(self, session: str) -> int:
        return self._orchestrator.get_chunk_count(session)
    
    def get_indexed_count(self, session: str) -> int:
        return self._orchestrator.get_indexed_count(session)
    
    def get_indexer(self) -> SRagIndexer:
        return self._orchestrator.get_indexer()

    def query(self, query, session, k=5, debug=False):
        return self._orchestrator.query(query, session, k=k, debug=debug)
    
    def query_session(self, query, session, k=5, debug=False):
        return self._orchestrator.query_session(query, session, k=k, debug=debug)
    
    def build_context(self, query, session, k=10, token_budget=None, debug=False):
        return self._orchestrator.build_context(query, session, k=k, token_budget=token_budget, debug=debug)

    def get_scraper(self) -> AnuInfrastructureScraper:
        return self._orchestrator.get_scraper()

    # ── Cache management ──────────────────────────────────────────────────────
    def is_stale(self, session: str, max_age_hours: int = 24) -> bool:
        return self._orchestrator.is_stale(session, max_age_hours)

    def list_sessions(self) -> list:
        return self._orchestrator.list_sessions()

     # ── Config access ─────────────────────────────────────────────────────────
    @property
    def config(self) -> SRagConfig:
        return self._orchestrator.config