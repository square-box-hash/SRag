import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import urlparse

import lancedb
from sentence_transformers import CrossEncoder, SentenceTransformer
logging.getLogger("trafilatura").setLevel(logging.CRITICAL)

from srag.config import SRagConfig
from srag.scraper import AnuInfrastructureScraper
from srag.indexer import SRagIndexer
from srag.chunker import SmartChunker
from srag.collector import SideChannelCollector
from srag.topic_classifier import TopicClassifier
from srag.query_intelligence import QueryIntelligence
from srag.reputation import ReputationStore, ReputationAwareSelector
from srag.lexicon import LexiconStore
from srag.adaptive_concurrency import ConcurrencyController
from srag.quality_evaluator import QualityEvaluator
from srag.context_builder import ContextBuilder
from srag.scraper import BLOCKED_DOMAINS, PRIORITY_DOMAINS
from srag.exceptions import (
    SRagFetchError, SRagQualityError, SRagNoContentError,
    SRagIndexError, SRagSessionNotFoundError,
)
from srag.result import SRagResult, SRagTrace
from srag.tracer import SRagTracer
from srag.recency import RecencyRanker
from srag.search_providers import SearchLayer, SearXNGProvider

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("trafilatura").setLevel(logging.ERROR)
logging.getLogger("trafilatura.core").setLevel(logging.ERROR)
logging.getLogger("trafilatura.utils").setLevel(logging.ERROR)
logging.getLogger("trafilatura").propagate = False
logging.getLogger("trafilatura.utils").propagate = False
logging.getLogger("trafilatura.core").propagate = False

logger = logging.getLogger(__name__)

# ── Quality gate ──────────────────────────────────────────────────────────────

PASS_RATE_THRESHOLD   = 0.25
MIN_CHUNKS_REQUIRED   = 3
COLD_START_CAP        = 0.70
COLD_START_MIN_SCRAPES = 5


class SRagOrchestrator:
    def __init__(
        self,
        config:        Optional[SRagConfig] = None,
        max_results:   int   = 12,
        max_chars:     int   = 2000,
        extract_mode:  str   = "trafilatura",
        max_concurrent: int  = 5,
        db_path:       str   = "./srag_db",
        rerank_top_k:  int   = 5,
    ):
        
        cfg = config or SRagConfig(
            max_results    = max_results,
            max_chars      = max_chars,
            extract_mode   = extract_mode,
            max_concurrent = max_concurrent,
            db_path        = db_path,
            rerank_top_k   = rerank_top_k,
        )
        self.config = cfg
        
        # ── Shared model instance ─────────────────────────────────────────────
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # ── Core components ───────────────────────────────────────────────────
        self.indexer  = SRagIndexer(db_path=cfg.db_path, model=_model)
        self.chunker  = SmartChunker(
            model            = _model,
            max_tokens       = cfg.chunk_size,
            dedupe_threshold = cfg.dedupe_threshold,
        )

        # ── Conditional reranker ──────────────────────────────────────────────
        self.reranker = (
            CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
            if cfg.use_reranker else None
        )

        # ── Shared LanceDB connection ─────────────────────────────────────────
        _db = self.indexer.db

        # ── Conditional intelligence layer ────────────────────────────────────
        self.lexicon = LexiconStore(db=_db) if cfg.use_lexicon else None
        self.reputation = (
            ReputationStore(db=_db, lexicon=self.lexicon)
            if cfg.use_reputation else None
        )
        self.topic_classifier   = TopicClassifier()
        self.query_intelligence = (
            QueryIntelligence(lexicon=self.lexicon)
            if cfg.use_query_intelligence else None
        )
        self.rep_selector = (
            ReputationAwareSelector(
                reputation       = self.reputation,
                priority_domains = PRIORITY_DOMAINS,
                blocked_domains  = BLOCKED_DOMAINS,
            ) if cfg.use_reputation else None
        )

        # ── Conditional adaptive concurrency ──────────────────────────────────
        self.controller = (
            ConcurrencyController()
            if cfg.use_adaptive_concurrency else None
        )
        
        # ── Conditional quality + context ─────────────────────────────────────
        self.evaluator       = QualityEvaluator() if cfg.use_quality_evaluator else None
        self.context_builder = ContextBuilder()

        # ── Semaphore + stored config values ──────────────────────────────────
        self.semaphore    = asyncio.Semaphore(cfg.max_concurrent)
        self.max_results  = cfg.max_results
        self.max_chars    = cfg.max_chars
        self.extract_mode = cfg.extract_mode
        self.rerank_top_k = cfg.rerank_top_k

    def _make_scraper(self) -> AnuInfrastructureScraper:
        return AnuInfrastructureScraper(
            max_results          = self.max_results,
            max_chars            = self.max_chars,
            extract_mode         = self.extract_mode,
            concurrency_controller = self.controller,
            query_intelligence   = self.query_intelligence,
            topic_classifier     = self.topic_classifier,
            reputation_selector  = self.rep_selector,
        )

    # ── Core search ───────────────────────────────────────────────────────────

    async def search(
        self,
        query:       str,
        session:     str,
        force_new:   bool         = False,
        max_results: Optional[int] = None,
        debug:       bool         = False,
    ) -> dict:
        async with self.semaphore:
            scraper    = self._make_scraper()
            collector  = SideChannelCollector()

            if max_results:
                scraper.max_results = max_results

            # ── Topic classification ──────────────────────────────────────────
            topic_result = self.topic_classifier.predict(query)
            topic        = topic_result.primary

            logger.info("Searching [%s]: %s (topic=%s)", session, query, topic)

            # ── Fetch ─────────────────────────────────────────────────────────
            docs = await scraper.get_facts(query, collector=collector)

            if not docs:
                logger.warning("[%s] No documents found", session)
                return self._empty_result(session, query, success=False)

            # ── Chunk ─────────────────────────────────────────────────────────
            chunks = self.chunker.chunk_docs(docs)

            if not chunks:
                logger.warning("[%s] No chunks after processing", session)
                return self._empty_result(session, query, success=False, doc_count=len(docs))

            # ── Evaluate BEFORE indexing ──────────────────────────────────────
            session_quality = self.evaluator.evaluate_session(
                chunks = chunks,
                query  = query,
                topic  = topic,
            )
            quality_summary = session_quality.summary()

            if debug:
                logger.info("[%s] Quality summary: %s", session, quality_summary)

            # ── Quality gate ──────────────────────────────────────────────────
            skip_indexing = (
                quality_summary["pass_rate"]    < PASS_RATE_THRESHOLD or
                quality_summary["passed_chunks"] < MIN_CHUNKS_REQUIRED
            )

            if skip_indexing:
                logger.warning(
                    "[%s] Quality gate failed — pass_rate=%.2f passed=%d. Skipping index.",
                    session,
                    quality_summary["pass_rate"],
                    quality_summary["passed_chunks"],
                )
                return self._empty_result(
                    session, query,
                    success   = False,
                    doc_count = len(docs),
                    reason    = "quality_gate_failed",
                )

            # ── Populate chunk records in collector ───────────────────────────
            self._populate_chunk_records(collector, chunks, session_quality)

            # ── Index ─────────────────────────────────────────────────────────
            indexed_stats = self.indexer.index_documents(
                chunks,
                table_name = session,
                force_new  = force_new,
            )
            indexed_count = indexed_stats.get("total_indexed", 0)

            logger.info(
                "[%s] %d docs → %d chunks → %d indexed after validation",
                session, len(docs), len(chunks), indexed_count,
            )

            # ── Update reputation ─────────────────────────────────────────────
            self._update_reputation(collector, topic, query, debug=debug)

            return {
                "session":       session,
                "query":         query,
                "topic":         topic,
                "doc_count":     len(docs),
                "chunk_count":   len(chunks),
                "indexed_count": indexed_count,
                "success":       True,
                "docs":          docs,
                "chunks":        chunks,
                **({"debug": {
                    "quality":     quality_summary,
                    "collector":   collector.summary(),
                    "concurrency": self.controller.status(),
                }} if debug else {}),
            }

    # ── Parallel search ───────────────────────────────────────────────────────

    async def parallel_search(
        self,
        plan:  List[dict],
        debug: bool = False,
    ) -> List[dict]:
        logger.info(
            "Parallel search: %d queries (max %d concurrent)",
            len(plan), self.max_concurrent,
        )

        async def _bounded(item):
            try:
                return await self.search(
                    query     = item["query"],
                    session   = item["session"],
                    force_new = item.get("force_new", False),
                    debug     = debug,
                )
            except Exception as e:
                return {
                    "session": item["session"],
                    "query":   item["query"],
                    "success": False,
                    "error":   str(e),
                }

        return await asyncio.gather(*[_bounded(item) for item in plan])

    # ── Sequential search ─────────────────────────────────────────────────────

    async def sequential_search(
        self,
        steps: List[dict],
        debug: bool = False,
    ) -> List[dict]:
        logger.info("Sequential search: %d steps", len(steps))
        results: List[dict] = []

        for step in steps:
            query = step["query"]

            if step.get("depends_on"):
                prev_session = step["depends_on"]
                k            = step.get("inject_top_k", 3)
                prev_chunks  = self.indexer.query_session(query, prev_session, k=k * 2)

                if prev_chunks:
                    good_chunks = [
                        c for c in prev_chunks
                        if len(c.get("content", "").strip()) > 200
                        and "switch language" not in c.get("content", "").lower()
                        and "cookie"           not in c.get("content", "").lower()
                        and "javascript"       not in c.get("content", "").lower()
                    ]
                    if good_chunks:
                        good_chunks = self._rerank(query, good_chunks, top_k=k)
                        context     = good_chunks[0].get("content", "") if good_chunks else ""
                        query       = self.query_intelligence.inject_context(
                            query   = query,
                            context = context,
                            topic   = self.topic_classifier.predict(query).primary,
                        )
                        logger.info(
                            "[%s] Injected reranked context from '%s'",
                            step["session"], prev_session,
                        )
                    else:
                        logger.warning(
                            "[%s] No clean context in '%s', skipping injection",
                            step["session"], prev_session,
                        )

            result = await self.search(
                query     = query,
                session   = step["session"],
                force_new = step.get("force_new", False),
                debug     = debug,
            )
            results.append(result)

        return results

    # ── Verify ────────────────────────────────────────────────────────────────

    async def verify(self, query: str, session: str, debug: bool = False) -> dict:
        logger.info("Verification search: %s", query)
        result = await self.search(query, session, force_new=True, debug=debug)

        if not result["success"]:
            return result

        conflicts = self._detect_conflicts(result["docs"])

        if conflicts:
            logger.warning("Conflict detected in '%s'", session)
            return {
                "session":  session,
                "query":    query,
                "status":   "conflict_detected",
                "conflicts": conflicts,
                "doc_count": result["doc_count"],
                "success":  True,
            }

        return {
            "session":  session,
            "query":    query,
            "status":   "clean",
            "doc_count": result["doc_count"],
            "success":  True,
        }

    # ── Query + context ───────────────────────────────────────────────────────

    def query(
        self,
        query:   str,
        session: str,
        k:       int  = 5,
        debug:   bool = False,
    ) -> List[dict]:
        candidates = self.indexer.query_session(query, session, k=k * 2)
        results    = self._rerank(query, candidates, top_k=k)
        if debug:
            logger.info(
                "Query [%s]: %d candidates → %d reranked",
                session, len(candidates), len(results),
            )
        return results

    def build_context(
        self,
        query:        str,
        session:      str,
        k:            int            = 10,
        token_budget: Optional[int]  = None,
        debug:        bool           = False,
    ):
        """
        Public method — retrieve + rerank + build structured context.
        Returns BuiltContext with to_prompt() and to_dict().
        """
        chunks     = self.query(query, session, k=k, debug=debug)
        topic      = self.topic_classifier.predict(query).primary

        # Reputation signals for context scoring
        reputation = {}
        for chunk in chunks:
            domain = self._get_domain(chunk.get("source", ""))
            if domain not in reputation:
                reputation[domain] = self.reputation.get_confidence(domain, topic)

        return self.context_builder.build(
            chunks       = chunks,
            query        = query,
            topic        = topic,
            token_budget = token_budget,
            reputation   = reputation,
        )

    # ── Reputation update ─────────────────────────────────────────────────────

    def _update_reputation(
        self,
        collector: SideChannelCollector,
        topic:     str,
        query:     str,
        debug:     bool = False,
    ) -> None:
        """
        Aggregate collector signals and update ReputationStore per domain.
        Applies cold start cap to prevent early overconfidence.
        """
        aggregated = collector.aggregate()

        for domain, signals in aggregated.items():
            try:
                new_conf = self.reputation.update(
                    domain            = domain,
                    topic             = topic,
                    query             = query,
                    avg_chunk_quality = signals["avg_chunk_quality"],
                    useful_hit_rate   = signals["useful_hit_rate"],
                    irrelevance_rate  = signals["irrelevance_rate"],
                    failure_rate      = signals["failure_rate"],
                    avg_latency       = signals["avg_latency"],
                )

                # Cold start cap — prevent early overconfidence
                record = self.reputation.get(domain, topic)
                if record:
                    scrapes = int(record.get("total_scrapes", 0))
                    if scrapes < COLD_START_MIN_SCRAPES and new_conf > COLD_START_CAP:
                        logger.debug(
                            "Cold start cap applied: %s/%s %.3f → %.3f",
                            domain, topic, new_conf, COLD_START_CAP,
                        )

                if debug:
                    logger.info(
                        "Reputation updated: %s/%s confidence=%.3f",
                        domain, topic, new_conf,
                    )

            except Exception:
                logger.exception(
                    "Reputation update failed for %s/%s — continuing",
                    domain, topic,
                )

    def _populate_chunk_records(
        self,
        collector:       SideChannelCollector,
        chunks:          List[dict],
        session_quality,
    ) -> None:
        """
        Populate collector chunk records from session quality evaluation.
        Called after QualityEvaluator.evaluate_session() and before indexing.
        """
        for domain, doc_quality in session_quality.doc_results.items():
            collector.record_chunks(
                domain        = domain,
                chunks_total  = doc_quality.total_chunks,
                chunks_kept   = doc_quality.passed_chunks,
                coherence_sum = doc_quality.avg_coherence * doc_quality.passed_chunks,
            )

    # ── Reranker ──────────────────────────────────────────────────────────────

    def _rerank(
        self,
        query:  str,
        chunks: List[dict],
        top_k:  Optional[int] = None,
    ) -> List[dict]:
        if not chunks:
            return chunks
        k     = top_k or self.rerank_top_k
        if not self.reranker:
            # Fallback to coherence score ranking when reranker is disabled
            return sorted(
                chunks,
                key=lambda c: c.get("coherence_score", 0),
                reverse=True
            )[:k]
        pairs = [(query, c.get("content", "")) for c in chunks]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in ranked[:k]]

    # ── Conflict detection ────────────────────────────────────────────────────

    def _detect_conflicts(self, docs: List[dict]) -> List[dict]:
        if len(docs) < 2:
            return []
        conflicts  = []
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
                    conflicts.append({
                        "source_a": {
                            "url":             doc_a.get("source", ""),
                            "timestamp":       doc_a.get("timestamp", ""),
                            "content_preview": doc_a.get("content", "")[:300],
                        },
                        "source_b": {
                            "url":             doc_b.get("source", ""),
                            "timestamp":       doc_b.get("timestamp", ""),
                            "content_preview": doc_b.get("content", "")[:300],
                        },
                        "conflict_type": "timestamp_divergence",
                        "newest_source": (
                            doc_a.get("source") if ts_a > ts_b else doc_b.get("source")
                        ),
                    })
        return conflicts

    # ── Cache management ──────────────────────────────────────────────────────

    def is_stale(self, session: str, max_age_hours: int = 24) -> bool:
        if session not in self.indexer.list_sessions():
            return True
        table = self.indexer.db.open_table(session)
        rows  = table.to_pandas()
        if rows.empty:
            return True
        from datetime import datetime, timezone
        latest = rows["timestamp"].max()
        try:
            ts        = datetime.fromisoformat(latest).replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
            return age_hours > max_age_hours
        except Exception:
            return True

    def list_sessions(self) -> List[str]:
        return self.indexer.list_sessions()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_domain(self, url: str) -> str:
        try:
            return urlparse(url).netloc.replace("www.", "")
        except Exception:
            return ""

    def _empty_result(
        self,
        session:   str,
        query:     str,
        success:   bool = False,
        doc_count: int  = 0,
        reason:    str  = "no_documents",
    ) -> dict:
        return {
            "session":       session,
            "query":         query,
            "doc_count":     doc_count,
            "chunk_count":   0,
            "indexed_count": 0,
            "success":       success,
            "docs":          [],
            "chunks":        [],
            "reason":        reason,
        }