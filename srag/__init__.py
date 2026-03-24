from dotenv import load_dotenv
load_dotenv()

from srag.orchestrator import SRagOrchestrator
from srag.scraper import AnuInfrastructureScraper
from srag.indexer import SRagIndexer


class SRag:
    """
    Unified public API for SRag.
    This is the only import Anu (or any user) needs.
    """
    def __init__(
        self,
        db_path: str = "./srag_db",
        max_results: int = 12,
        max_chars: int = 2000,
        extract_mode: str = "trafilatura",
        max_concurrent: int = 5,
    ):
        self._orchestrator = SRagOrchestrator(
            max_results=max_results,
            max_chars=max_chars,
            extract_mode=extract_mode,
            max_concurrent=max_concurrent,
            db_path=db_path,
        )

    # ── Search modes ──────────────────────────────────────────────────────────
    async def search(self, query: str, session: str, force_new: bool = False) -> dict:
        return await self._orchestrator.search(query, session, force_new)

    async def parallel_search(self, plan: list) -> list:
        return await self._orchestrator.parallel_search(plan)

    async def sequential_search(self, steps: list) -> list:
        return await self._orchestrator.sequential_search(steps)

    async def verify(self, query: str, session: str) -> dict:
        return await self._orchestrator.verify(query, session)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    def query(self, question: str, session: str, k: int = 5) -> list:
        return self._orchestrator.query(question, session, k)

    # ── Cache management ──────────────────────────────────────────────────────
    def is_stale(self, session: str, max_age_hours: int = 24) -> bool:
        return self._orchestrator.is_stale(session, max_age_hours)

    def list_sessions(self) -> list:
        return self._orchestrator.list_sessions()