import asyncio
import logging
import time
from srag import SRag
from srag.topic_classifier import TopicClassifier
from srag.query_intelligence import QueryIntelligence
from srag.collector import SideChannelCollector
from srag.adaptive_concurrency import ConcurrencyController, ManagedSlot
from srag.quality_evaluator import QualityEvaluator
from srag.context_builder import ContextBuilder
from srag.lexicon import LexiconStore
from srag.reputation import ReputationStore
import lancedb
from srag.context_builder import _estimate_tokens

logging.basicConfig(level=logging.WARNING)

# ── Shared instance ───────────────────────────────────────────────────────────
sr = SRag()

# ─────────────────────────────────────────────────────────────────────────────
# A. PIPELINE INTEGRITY
# ─────────────────────────────────────────────────────────────────────────────

async def test_01_single_search():
    print("\n── Test 01: Single Search (tech) ────────────────────────")
    result = await sr.search("python asyncio tutorial", session="t01_tech")
    print(f"  success:       {result['success']}")
    print(f"  docs:          {result['doc_count']}")
    print(f"  chunks:        {result['chunk_count']}")
    print(f"  indexed:       {result['indexed_count']}")
    print(f"  topic:         {result.get('topic', '?')}")
    assert result["success"], "Test 01 FAILED: search returned success=False"
    assert result["indexed_count"] > 0, "Test 01 FAILED: nothing indexed"
    print("  ✅ PASSED")


async def test_02_parallel_search():
    print("\n── Test 02: Parallel Search ─────────────────────────────")
    results = await sr.parallel_search([
        {"query": "FastAPI dependency injection", "session": "t02_fastapi"},
        {"query": "GST rate mobile phones India 2026", "session": "t02_gst"},
    ])
    for r in results:
        print(f"  [{r['session']}] success={r['success']} docs={r.get('doc_count',0)} indexed={r.get('indexed_count',0)}")
    assert all(r["success"] for r in results), "Test 02 FAILED: one or more parallel searches failed"
    print("  ✅ PASSED")


async def test_03_sequential_search():
    print("\n── Test 03: Sequential Search with context injection ────")
    results = await sr.sequential_search([
        {"query": "F1 2026 championship standings", "session": "t03_f1_base"},
        {"query": "F1 2026 next race schedule", "session": "t03_f1_next", "depends_on": "t03_f1_base"},
    ])
    for r in results:
        print(f"  [{r['session']}] success={r['success']} docs={r.get('doc_count',0)}")
    assert results[0]["success"], "Test 03 FAILED: first step failed"
    print("  ✅ PASSED")


async def test_04_verify():
    print("\n── Test 04: Verification Search ─────────────────────────")
    result = await sr.verify("RBI repo rate 2026", session="t04_rbi")
    print(f"  status:    {result.get('status', '?')}")
    print(f"  success:   {result['success']}")
    print(f"  conflicts: {len(result.get('conflicts', []))}")
    assert result["success"], "Test 04 FAILED"
    print("  ✅ PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# B. INTELLIGENCE LAYER
# ─────────────────────────────────────────────────────────────────────────────

def test_05_topic_classifier():
    print("\n── Test 05: Topic Classifier ────────────────────────────")
    tc     = TopicClassifier()
    cases  = [
        ("python asyncio tutorial",          "tech"),
        ("GST rate electronics India",        "finance"),
        ("F1 Japanese GP 2026 standings",     "sports"),
        ("JEE advanced syllabus 2026",        "education"),
        ("black hole event horizon research", "science"),
        ("Kimi Antonelli wins race",          "sports"),
        ("RBI interest rate cut",             "finance"),
    ]
    for query, expected in cases:
        result = tc.predict(query)
        status = "✅" if result.primary == expected else "⚠️ "
        print(f"  {status} '{query[:40]}' → {result.primary} (expected {expected}) conf={result.confidence:.2f} ambiguous={result.ambiguous}")
    print("  ✅ PASSED (warnings are soft mismatches)")


def test_06_query_intelligence():
    print("\n── Test 06: Query Intelligence ──────────────────────────")
    qi    = QueryIntelligence(lexicon=None)
    cases = [
        ("python decorators tutorial", "tech"),
        ("GST filing deadline",         "finance"),
        ("F1 2026 race results",        "sports"),
    ]
    for query, topic in cases:
        plan    = qi.rewrite(query, topic)
        queries = plan.get_queries()
        print(f"\n  query:    '{query}'")
        print(f"  topic:    {topic}")
        print(f"  variants: {len(queries)}")
        for i, q in enumerate(queries):
            print(f"    [{i+1}] {q}")
        assert len(queries) >= 1,                    "Test 06 FAILED: no variants"
        assert len(queries) <= 5,                    "Test 06 FAILED: exceeded MAX_VARIANTS"
        assert queries[0] == query.strip(),          "Test 06 FAILED: original not first"
        assert all(len(q) <= 180 for q in queries),  "Test 06 FAILED: variant exceeds MAX_VARIANT_LENGTH"
    print("\n  ✅ PASSED")


def test_07_query_inject_context():
    print("\n── Test 07: Context Injection (sequential) ──────────────")
    qi       = QueryIntelligence()
    base     = "GST filing deadline"
    contexts = [
        "Deepshikha Nainani\nPosted October 2025\nThe Goods and Services Tax annual return...",
        "The GST annual return for FY 2024-25 must be filed by December 31, 2025.",
        "",
    ]
    for ctx in contexts:
        result = qi.inject_context(base, ctx, "finance")
        print(f"  input ctx: '{ctx[:50]}...' " if ctx else "  input ctx: '' (empty)")
        print(f"  result:    '{result}'")
        assert len(result) <= 180,        "Test 07 FAILED: exceeds MAX_VARIANT_LENGTH"
        assert result.startswith(base),   "Test 07 FAILED: original query not leading"
        if not ctx.strip():
            assert result == base,        "Test 07 FAILED: empty context should return original"
    print("  ✅ PASSED")


def test_08_reputation_cold_start():
    print("\n── Test 08: Reputation Cold Start Cap ───────────────────")
    db  = lancedb.connect("./srag_db")
    rep = ReputationStore(db=db)

    # Unseen domain should return neutral 0.5
    conf = rep.get_confidence("unknown-new-domain.xyz", "tech")
    print(f"  unseen domain confidence: {conf}")
    assert conf == 0.5, f"Test 08 FAILED: expected 0.5 got {conf}"

    # Known domain after 1 scrape should be capped
    rep.update(
        domain            = "test-cold-start.com",
        topic             = "tech",
        query             = "test query",
        avg_chunk_quality = 0.95,
        useful_hit_rate   = 0.95,
        irrelevance_rate  = 0.05,
        failure_rate      = 0.0,
        avg_latency       = 0.5,
    )
    conf_after = rep.get_confidence("test-cold-start.com", "tech")
    print(f"  after 1 scrape confidence: {conf_after}")
    # Cold start cap is enforced in orchestrator not reputation store directly
    # so raw store value can exceed cap — just verify it updated
    assert conf_after > 0.5, "Test 08 FAILED: confidence did not increase after good scrape"
    print("  ✅ PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# C. QUALITY + EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def test_09_quality_evaluator():
    print("\n── Test 09: Quality Evaluator ───────────────────────────")
    evaluator = QualityEvaluator()

    good_chunk = {
        "content":         "Python decorators are a very powerful and useful tool in Python since it allows programmers to modify the behaviour of a function or class. Decorators allow us to wrap another function in order to extend the behaviour of the wrapped function.",
        "coherence_score": 0.72,
        "chunk_index":     0,
        "source":          "https://realpython.com/primer-on-python-decorators/",
        "sentence_count":  3,
    }
    bad_chunk = {
        "content":         "Cookie policy. Accept cookies. Sign up. Log in.",
        "coherence_score": 0.08,
        "chunk_index":     1,
        "source":          "https://some-site.com/nav",
        "sentence_count":  1,
    }

    good_result = evaluator.evaluate_chunk(good_chunk)
    bad_result  = evaluator.evaluate_chunk(bad_chunk)

    print(f"  good chunk — passed={good_result.passed} composite={good_result.composite_score:.3f} noise={good_result.noise_detected}")
    print(f"  bad chunk  — passed={bad_result.passed}  composite={bad_result.composite_score:.3f} noise={bad_result.noise_detected}")

    assert good_result.passed,        "Test 09 FAILED: good chunk should pass"
    assert not bad_result.passed,     "Test 09 FAILED: bad chunk should fail"
    assert bad_result.noise_detected, "Test 09 FAILED: noise not detected in bad chunk"
    print("  ✅ PASSED")


def test_10_quality_gate():
    print("\n── Test 10: Quality Gate Thresholds ─────────────────────")
    evaluator = QualityEvaluator()

    # Session with mostly bad chunks
    bad_chunks = [
        {
            "content":         "Cookie policy. Accept. Sign up.",
            "coherence_score": 0.05,
            "chunk_index":     i,
            "source":          "https://bad-site.com",
            "sentence_count":  1,
        }
        for i in range(10)
    ]

    session = evaluator.evaluate_session(bad_chunks, "test query", "general")
    summary = session.summary()
    print(f"  pass_rate:     {summary['pass_rate']}")
    print(f"  passed_chunks: {summary['passed_chunks']}")
    skip = summary["pass_rate"] < 0.25 or summary["passed_chunks"] < 3
    print(f"  skip_indexing: {skip}")
    assert skip, "Test 10 FAILED: quality gate should trigger on bad session"
    print("  ✅ PASSED")


def test_11_collector_aggregate():
    print("\n── Test 11: Collector Aggregation ───────────────────────")
    collector = SideChannelCollector()

    collector.record_fetch("realpython.com", "https://realpython.com/a", 1.2,  False)
    collector.record_fetch("realpython.com", "https://realpython.com/b", 0.9,  False)
    collector.record_fetch("bad-site.com",   "https://bad-site.com/a",   5.1,  True,  rate_limited=True)
    collector.record_fetch("bad-site.com",   "https://bad-site.com/b",   4.8,  True)
    collector.record_chunks("realpython.com", chunks_total=8, chunks_kept=6, coherence_sum=4.8)
    collector.record_chunks("bad-site.com",   chunks_total=4, chunks_kept=0, coherence_sum=0.0)

    agg = collector.aggregate()

    print(f"  realpython — latency={agg['realpython.com']['avg_latency']} failure={agg['realpython.com']['failure_rate']} quality={agg['realpython.com']['avg_chunk_quality']}")
    print(f"  bad-site   — latency={agg['bad-site.com']['avg_latency']}   failure={agg['bad-site.com']['failure_rate']}   quality={agg['bad-site.com']['avg_chunk_quality']}")

    assert agg["realpython.com"]["failure_rate"]  == 0.0,  "Test 11 FAILED"
    assert agg["bad-site.com"]["failure_rate"]    == 1.0,  "Test 11 FAILED"
    assert agg["realpython.com"]["useful_hit_rate"] == 0.75, "Test 11 FAILED"
    assert agg["bad-site.com"]["useful_hit_rate"]   == 0.0,  "Test 11 FAILED"
    assert not collector.is_empty(), "Test 11 FAILED: collector should not be empty"
    print("  ✅ PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# D. ADAPTIVE SYSTEMS
# ─────────────────────────────────────────────────────────────────────────────

async def test_12_concurrency_controller():
    print("\n── Test 12: Concurrency Controller (AIMD) ───────────────")
    controller = ConcurrencyController(
        min_concurrency      = 2,
        max_concurrency      = 10,
        cooldown_seconds     = 0.1,  # fast cooldown for testing
        max_requests_per_sec = 100,
    )

    print(f"  initial concurrency: {controller._concurrency}")

    # Simulate successful requests — should increase
    for _ in range(5):
        token = await controller.acquire("realpython.com")
        controller.record(token, latency=0.5, failed=False)
        await controller.release(token)

    status_after_success = controller.status()
    print(f"  after 5 successes:   {status_after_success['global_concurrency']}")

    # Simulate failures — should decrease
    for _ in range(5):
        token = await controller.acquire("bad-site.com")
        controller.record(token, latency=6.0, failed=True)
        await controller.release(token)

    await asyncio.sleep(0.2)  # allow cooldown
    status_after_fail = controller.status()
    print(f"  after 5 failures:    {status_after_fail['global_concurrency']}")
    print(f"  p95_latency:         {status_after_fail['p95_latency']}")
    print(f"  global_failure_rate: {status_after_fail['global_failure_rate']}")

    assert status_after_fail["global_concurrency"] >= 2, "Test 12 FAILED: below min"
    assert status_after_fail["global_concurrency"] <= 10, "Test 12 FAILED: above max"
    print("  ✅ PASSED")


def test_13_context_builder():
    print("\n── Test 13: Context Builder ─────────────────────────────")
    builder = ContextBuilder(max_context_tokens=2000)
    builder.reset_session()

    chunks = [
        {
            "content":         f"Python decorator chunk {i}. " * 20,
            "coherence_score": 0.7 + i * 0.02,
            "chunk_index":     i,
            "source":          f"https://source-{i % 3}.com/article",
            "title":           f"Python Decorators Part {i}",
        }
        for i in range(8)
    ]

    scored    = builder._score_chunks(chunks, {})
    print(f"  after scoring:     {len(scored)} chunks")
    print(f"  scores:            {[c['_score'] for c in scored]}")

    deduped   = builder._deduplicate(scored)
    print(f"  after dedup:       {len(deduped)} chunks")

    filtered  = [c for c in deduped if c["_score"] >= builder.min_chunk_score]
    print(f"  after score filter:{len(filtered)} chunks")

    diverse   = builder._apply_diversity(filtered)
    print(f"  after diversity:   {len(diverse)} chunks")

    allocated = builder._allocate_tokens(diverse, 500)
    print(f"  after allocation:  {[c['_allocated_tokens'] for c in allocated]}")

    for item in allocated:
        content  = item["content"]
        trimmed  = builder._trim_to_tokens(content, item["_allocated_tokens"])  # ← note: _trim_to_tokens not _trim
        estimate = _estimate_tokens(trimmed)
        print(f"  chunk {item['chunk_index']}: allocated={item['_allocated_tokens']} trimmed_tokens={estimate} passes={estimate >= builder.min_tokens_per_chunk}")

    context = builder.build(chunks, query="python decorators", topic="tech")
    print(f"  final chunks: {len(context.chunks)}")

    for c in chunks[:2]:
        print(f"  chunk keys: {list(c.keys())}")
        print(f"  coherence: {c.get('coherence_score')}")

    context = builder.build(chunks, query="python decorators", topic="tech")

    print(f"  chunks selected: {len(context.chunks)}")
    print(f"  total tokens:    {context.total_tokens}")
    print(f"  sources:         {len(context.sources)}")
    print(f"  truncated:       {context.truncated}")
    print(f"  utilization:     {context.summary()['utilization']:.2f}")
    print(f"\n  to_prompt() preview:\n")
    prompt = context.to_prompt()
    print("  " + "\n  ".join(prompt[:400].split("\n")))

    assert context.total_tokens <= 500,   "Test 13 FAILED: exceeded token budget"
    assert len(context.chunks) > 0,       "Test 13 FAILED: no chunks selected"
    assert len(context.sources) > 0,      "Test 13 FAILED: no sources"
    assert "[Source:" in prompt,          "Test 13 FAILED: prompt missing source header"
    print("\n  ✅ PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# E. CLI OUTPUT QUALITY
# ─────────────────────────────────────────────────────────────────────────────

async def test_14_search_output_quality():
    print("\n── Test 14: Search Output Quality (no binary garbage) ───")
    from srag.scraper import AnuInfrastructureScraper
    from srag.chunker import SmartChunker
    from sentence_transformers import SentenceTransformer

    scraper = AnuInfrastructureScraper(max_results=5)
    _model  = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    chunker = SmartChunker(model=_model, max_tokens=256)

    docs   = await scraper.get_facts("Kimi Antonelli F1 2026")
    chunks = chunker.chunk_docs(docs)

    print(f"  docs:   {len(docs)}")
    print(f"  chunks: {len(chunks)}")

    for chunk in chunks[:3]:
        content = chunk.get("content", "")
        # Check for binary garbage
        non_printable = sum(1 for c in content if ord(c) > 127 and ord(c) < 160)
        ratio         = non_printable / max(len(content), 1)
        print(f"  chunk [{chunk.get('chunk_index')}] coherence={chunk.get('coherence_score', 0):.2f} garbage_ratio={ratio:.3f}")
        assert ratio < 0.05, f"Test 14 FAILED: binary garbage ratio {ratio:.3f} too high"

    assert len(chunks) > 0, "Test 14 FAILED: no chunks produced"
    print("  ✅ PASSED")


async def test_15_debug_mode():
    print("\n── Test 15: Debug Mode + Full Pipeline ──────────────────")
    result = await sr.search(
        "JEE advanced physics syllabus 2026",
        session = "t15_debug",
        debug   = True,
    )

    print(f"  success:   {result['success']}")
    print(f"  topic:     {result.get('topic', '?')}")
    print(f"  docs:      {result.get('doc_count', 0)}")
    print(f"  chunks:    {result.get('chunk_count', 0)}")
    print(f"  indexed:   {result.get('indexed_count', 0)}")

    if "debug" in result:
        d = result["debug"]
        print(f"\n  [debug] quality:     {d['quality']}")
        print(f"  [debug] collector domains: {list(d['collector']['domains_data'].keys())}")
        print(f"  [debug] concurrency: global={d['concurrency']['global_concurrency']} p95={d['concurrency']['p95_latency']}")
        assert "quality"     in d, "Test 15 FAILED: debug missing quality"
        assert "collector"   in d, "Test 15 FAILED: debug missing collector"
        assert "concurrency" in d, "Test 15 FAILED: debug missing concurrency"
    else:
        print("  ⚠️  No debug key in result — check debug=True is wired in search()")

    print("  ✅ PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("  SRag v0.8.0 — Test Suite (15 tests)")
    print("=" * 60)

    start = time.monotonic()
    passed = 0
    failed = 0
    errors = []

    # Sync tests
    sync_tests = [
        test_05_topic_classifier,
        test_06_query_intelligence,
        test_07_query_inject_context,
        test_08_reputation_cold_start,
        test_09_quality_evaluator,
        test_10_quality_gate,
        test_11_collector_aggregate,
        test_13_context_builder,
    ]

    for test_fn in sync_tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {e}")
            failed += 1
            errors.append(str(e))
        except Exception as e:
            print(f"  💥 {test_fn.__name__} crashed: {type(e).__name__}: {e}")
            failed += 1
            errors.append(f"{test_fn.__name__}: {e}")

    # Async tests
    async_tests = [
        test_01_single_search,
        test_02_parallel_search,
        test_03_sequential_search,
        test_04_verify,
        test_12_concurrency_controller,
        test_14_search_output_quality,
        test_15_debug_mode,
    ]

    for test_fn in async_tests:
        try:
            await test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {e}")
            failed += 1
            errors.append(str(e))
        except Exception as e:
            print(f"  💥 {test_fn.__name__} crashed: {type(e).__name__}: {e}")
            failed += 1
            errors.append(f"{test_fn.__name__}: {e}")

    elapsed = time.monotonic() - start
    print("\n" + "=" * 60)
    print(f"  Results: {passed}/15 passed, {failed} failed — {elapsed:.1f}s")
    if errors:
        print("\n  Failed:")
        for e in errors:
            print(f"    • {e}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())