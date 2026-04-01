import asyncio
from typing import List, Optional

from srag.scraper import AnuInfrastructureScraper
from srag.indexer import SRagIndexer


class SRagOrchestrator:
    def __init__(
        self,
        max_results: int = 12,
        max_chars: int = 2000,
        extract_mode: str = "trafilatura",
        max_concurrent: int = 5,
        db_path: str = "./srag_db",
    ):
        self.max_results = max_results
        self.max_chars = max_chars
        self.extract_mode = extract_mode
        self.max_concurrent = max_concurrent
        self.indexer = SRagIndexer(db_path=db_path)
        self.semaphore = asyncio.Semaphore(max_concurrent)

    def _make_scraper(self):
        return AnuInfrastructureScraper(
            max_results=self.max_results,
            max_chars=self.max_chars,
            extract_mode=self.extract_mode,
        )

    # ── Core single search ────────────────────────────────────────────────────
    async def search(
        self,
        query: str,
        session: str,
        force_new: bool = False,
        max_results: Optional[int] = None,
    ) -> dict:
        async with self.semaphore:
            scraper = self._make_scraper()
            if max_results:
                scraper.max_results = max_results

            print(f"🔍 [{session}] Searching: {query}")
            docs = await scraper.get_facts(query)

            if not docs:
                print(f"❌ [{session}] No documents found.")
                return {
                    "session": session,
                    "query": query,
                    "doc_count": 0,
                    "success": False,
                    "docs": [],
                }

            self.indexer.index_documents(docs, table_name=session, force_new=force_new)
            print(f"✅ [{session}] Indexed {len(docs)} docs.")

            return {
                "session": session,
                "query": query,
                "doc_count": len(docs),
                "success": True,
                "docs": docs,
            }

    # ── Parallel search ───────────────────────────────────────────────────────
    async def parallel_search(self, plan: List[dict]) -> List[dict]:
        """
        plan: [{"query": str, "session": str, "force_new": bool (optional)}]
        """
        print(
            f"\n🚀 Parallel search: {len(plan)} queries "
            f"(max {self.max_concurrent} concurrent)\n"
        )

        async def _bounded(item):
            try:
                return await self.search(
                    query=item["query"],
                    session=item["session"],
                    force_new=item.get("force_new", False),
                )
            except Exception as e:
                return {
                    "session": item["session"],
                    "query": item["query"],
                    "success": False,
                    "error": str(e),
                }

        return await asyncio.gather(*[_bounded(item) for item in plan])

    # ── Sequential search ─────────────────────────────────────────────────────
    async def sequential_search(self, steps: List[dict]) -> List[dict]:
        """
        steps: [
            {"query": str, "session": str},
            {"query": str, "session": str, "depends_on": str, "inject_top_k": int}
        ]
        depends_on: session name of previous step to inject context from
        inject_top_k: how many chunks from previous session to inject into next query
        """
        print(f"\n🔗 Sequential search: {len(steps)} steps\n")
        results: List[dict] = []

        for step in steps:
            query = step["query"]

            # If this step depends on a previous session, inject context from it
            if step.get("depends_on"):
                prev_session = step["depends_on"]
                k = step.get("inject_top_k", 3)
                prev_chunks = self.indexer.query_session(query, prev_session, k=k)

                if prev_chunks:
                    good_chunks = [
                        c
                        for c in prev_chunks
                        if len(c.get("content", "").strip()) > 200
                        and "switch language" not in c.get("content", "").lower()
                        and "cookie" not in c.get("content", "").lower()
                        and "javascript" not in c.get("content", "").lower()
                        and "enable javascript" not in c.get("content", "").lower()
                    ]
                    if good_chunks:
                        injected_context = " ".join(
                            [c["content"][:300] for c in good_chunks[:2]]
                        )
                        query = f"{query} context: {injected_context}"
                        print(
                            f"💉 [{step['session']}] Injected context from '{prev_session}'"
                        )
                    else:
                        print(
                            f"⚠️  [{step['session']}] No clean context in "
                            f"'{prev_session}', skipping injection"
                        )

            result = await self.search(
                query=query,
                session=step["session"],
                force_new=step.get("force_new", False),
            )
            results.append(result)

        return results

    # ── Verification search ───────────────────────────────────────────────────
    async def verify(self, query: str, session: str) -> dict:
        """
        Runs a fresh search and checks for conflicts between sources.
        Returns structured conflict JSON if detected, clean result otherwise.
        """
        print(f"\n🔎 Verification search: {query}\n")
        result = await self.search(query, session, force_new=True)

        if not result["success"]:
            return result

        docs = result["docs"]
        conflicts = self._detect_conflicts(docs)

        if conflicts:
            print(f"⚠️  Conflict detected in '{session}'")
            return {
                "session": session,
                "query": query,
                "status": "conflict_detected",
                "conflicts": conflicts,
                "doc_count": len(docs),
                "success": True,
            }

        return {
            "session": session,
            "query": query,
            "status": "clean",
            "doc_count": len(docs),
            "success": True,
        }

    def _detect_conflicts(self, docs: List[dict]) -> List[dict]:
        """
        Basic conflict detection — flags docs with significantly
        different timestamps on the same topic as potential conflicts.
        Supervisor handles resolution, SRag only surfaces.
        """
        if len(docs) < 2:
            return []

        conflicts: List[dict] = []
        seen_pairs = set()

        for i, doc_a in enumerate(docs):
            for j, doc_b in enumerate(docs):
                if i >= j:
                    continue

                pair_key = (i, j)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                ts_a = doc_a.get("timestamp", "")[:7]  # YYYY-MM
                ts_b = doc_b.get("timestamp", "")[:7]

                if ts_a and ts_b and ts_a != ts_b:
                    conflicts.append(
                        {
                            "source_a": {
                                "url": doc_a.get("source", ""),
                                "timestamp": doc_a.get("timestamp", ""),
                                "content_preview": doc_a.get("content", "")[:300],
                            },
                            "source_b": {
                                "url": doc_b.get("source", ""),
                                "timestamp": doc_b.get("timestamp", ""),
                                "content_preview": doc_b.get("content", "")[:300],
                            },
                            "conflict_type": "timestamp_divergence",
                            "newest_source": (
                                doc_a.get("source")
                                if ts_a > ts_b
                                else doc_b.get("source")
                            ),
                        }
                    )

        return conflicts

    # ── Cache management ──────────────────────────────────────────────────────
    def is_stale(self, session: str, max_age_hours: int = 24) -> bool:
        """
        Checks if a session's data is older than max_age_hours.
        Returns True if stale or session doesn't exist.
        """
        if session not in self.indexer.list_sessions():
            return True

        table = self.indexer.db.open_table(session)
        rows = table.to_pandas()

        if rows.empty:
            return True

        from datetime import datetime, timezone

        latest = rows["timestamp"].max()

        try:
            ts = datetime.fromisoformat(latest).replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
            return age_hours > max_age_hours
        except Exception:
            return True

    def list_sessions(self) -> List[str]:
        return self.indexer.list_sessions()

    def query(self, query: str, session: str, k: int = 5) -> List[dict]:
        return self.indexer.query_session(query, session, k=k)