import asyncio
from typing import List, Optional
import sentence_transformers
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

from srag.scraper import AnuInfrastructureScraper
from srag.indexer import SRagIndexer
from srag.chunker import SmartChunker


def _extract_core(text: str) -> str:
    # Skip past any author/date header lines
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 40]
    return lines[0][:80] if lines else text[:80]


class SRagOrchestrator:
    def __init__(
        self,
        max_results: int = 12,
        max_chars: int = 2000,
        extract_mode: str = "trafilatura",
        max_concurrent: int = 5,
        db_path: str = "./srag_db",
        rerank_top_k: int = 5,
    ):
        self.max_results = max_results
        self.max_chars = max_chars
        self.extract_mode = extract_mode
        self.max_concurrent = max_concurrent
        self.rerank_top_k = rerank_top_k

        # Use a single shared model instance
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        self.indexer = SRagIndexer(db_path=db_path, model=self.model)
        self.semaphore = asyncio.Semaphore(max_concurrent)

        self.chunker = SmartChunker(
            model=self.model,
            max_tokens=256
        )

        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,
        )


    def _rerank(self, query: str, chunks: List[dict], top_k: Optional[int] = None) -> List[dict]:
        if not chunks:
            return chunks

        k = top_k or self.rerank_top_k
        pairs = [(query, c.get("content", "")) for c in chunks]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in ranked[:k]]


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
            if max_results is not None:
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

            chunks = self.chunker.chunk_docs(docs)

            if not chunks:
                print(f"❌ [{session}] No chunks created from documents.")
                return {
                    "session": session,
                    "query": query,
                    "doc_count": len(docs),
                    "chunk_count": 0,
                    "success": False,
                    "docs": docs,
                }

            print(f"🧩 [{session}] {len(docs)} docs → {len(chunks)} chunks")

            # Capture how many were actually indexed (if your indexer returns a count)
            indexed_count = self.indexer.index_documents(
                chunks,
                table_name=session,
                force_new=force_new,
            )

            print(
                f"✅ [{session}] Indexed {len(chunks)} chunks "
                f"→ {indexed_count} indexed after coherence filter."
            )

            return {
                "session": session,
                "query": query,
                "doc_count": len(docs),
                "chunk_count": len(chunks),
                "success": True,
                "docs": docs,
                "chunks": chunks,
            }


    # ── Parallel search ───────────────────────────────────────────────────────
    async def parallel_search(self, plan: List[dict]) -> List[dict]:
        """
        plan: [
            {"query": str, "session": str, "force_new": bool (optional), "max_results": int (optional)}
        ]
        """
        print(
            f"\n🚀 Parallel search: {len(plan)} queries "
            f"(max {self.max_concurrent} concurrent)\n"
        )

        import traceback

        async def _bounded(item):
            try:
                return await self.search(
                    query=item["query"],
                    session=item["session"],
                    force_new=item.get("force_new", False),
                    max_results=item.get("max_results"),
                )
            except Exception as e:
                print(f"❌ [{item['session']}] Error: {e}\n{traceback.format_exc()}")
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
            {
                "query": str,
                "session": str,
                "depends_on": str,        # session name of previous step
                "inject_top_k": int       # how many chunks to inject
            }
        ]
        """
        print(f"\n🔗 Sequential search: {len(steps)} steps\n")
        results: List[dict] = []

        for step in steps:
            query = step["query"]

            if step.get("depends_on"):
                prev_session = step["depends_on"]
                k = step.get("inject_top_k", 3)
                prev_chunks = self.indexer.query_session(query, prev_session, k=k * 2)

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
                        good_chunks = self._rerank(query, good_chunks, top_k=k)
                        injected_context = _extract_core(good_chunks[0]["content"])
                        query = f"{query} {injected_context}"
                        print(
                            f"💉 [{step['session']}] Injected reranked context from '{prev_session}'"
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

                ts_a = doc_a.get("timestamp", "")[:7]
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
        if session not in self.indexer.list_sessions():
            return True

        table = self.indexer.db.open_table(session)
        rows = table.to_pandas()

        if rows.empty:
            return True

        from datetime import datetime, timezone

        latest = rows["timestamp"].max()

        try:
            dt = datetime.fromisoformat(latest)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
            return age_hours > max_age_hours
        except Exception:
            return True  # treat malformed / missing timestamps as stale


    def list_sessions(self) -> List[str]:
        return self.indexer.list_sessions()


    def query(self, query: str, session: str, k: int = 5) -> List[dict]:
        # Fetch more candidates than needed, then rerank down to k
        candidates = self.indexer.query_session(query, session, k=k * 2)
        return self._rerank(query, candidates, top_k=k)