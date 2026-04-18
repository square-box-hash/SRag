import asyncio
from srag import SRag

async def main():
    sr = SRag()

    # Test 1: single search
    print("\n--- Test 1: Single Search ---")
    result = await sr.search("python decorators tutorial", session="test_single")
    print(f"Success: {result['success']}, Docs: {result['doc_count']}, Chunks: {result['chunk_count']}")

    # Test 2: parallel search
    print("\n--- Test 2: Parallel Search ---")
    results = await sr.parallel_search([
        {"query": "FastAPI tutorial", "session": "test_fastapi"},
        {"query": "LanceDB tutorial", "session": "test_lancedb"},
    ])
    for r in results:
        print(f"{r['session']}: {r['doc_count']} docs, {r['chunk_count']} chunks")

    # Test 3: sequential search
    print("\n--- Test 3: Sequential Search ---")
    results = await sr.sequential_search([
        {"query": "GST rate India electronics", "session": "test_seq_1"},
        {"query": "GST filing deadline", "session": "test_seq_2", "depends_on": "test_seq_1"},
    ])
    for r in results:
        print(f"{r['session']}: {r['doc_count']} docs, {r['chunk_count']} chunks")

    # Test 4: query stored session
    print("\n--- Test 4: Query Session ---")
    chunks = sr.query("what is the GST rate", session="test_seq_1", k=3)
    for c in chunks:
        print(f"- {c.get('title', 'No Title')}: {c.get('content', '')[:100]}")

    # Test 5: is_stale
    print("\n--- Test 5: Cache Check ---")
    print(f"test_single stale: {sr.is_stale('test_single', max_age_hours=24)}")
    print(f"nonexistent stale: {sr.is_stale('nonexistent')}")

    # Test 6: list sessions
    print("\n--- Test 6: List Sessions ---")
    print(sr.list_sessions())


if __name__ == "__main__":
    asyncio.run(main())
