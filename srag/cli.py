import argparse
import asyncio
import json
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        prog="srag",
        description="SRag — Smart RAG CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  srag search "latest AI news"
  srag index "python asyncio tutorial" --session python_async
  srag index "GST rate India" --session gst --results 15 --force-new
  srag query "what is the rate" --session gst
  srag query "what is the rate" --session gst --k 10
  srag verify "GST rate India" --session gst_verify
  srag sessions
        """
    )

    sub = parser.add_subparsers(dest="command", metavar="command")

    # ── srag search ───────────────────────────────────────────────────────────
    s = sub.add_parser("search", help="Search the web and print results (no indexing)")
    s.add_argument("query", type=str, help="Search query")
    s.add_argument("--results", type=int, default=12, help="Number of results (default: 12)")

    # ── srag index ────────────────────────────────────────────────────────────
    idx = sub.add_parser("index", help="Search, scrape and index into a session")
    idx.add_argument("query", type=str, help="Search query")
    idx.add_argument("--session", type=str, required=True, help="Session name (table in LanceDB)")
    idx.add_argument("--results", type=int, default=12, help="Number of URLs to scrape (default: 12)")
    idx.add_argument("--force-new", action="store_true", help="Wipe and rebuild this session")

    # ── srag query ────────────────────────────────────────────────────────────
    q = sub.add_parser("query", help="Semantic search within a stored session")
    q.add_argument("query", type=str, help="Question to search for")
    q.add_argument("--session", type=str, required=True, help="Session name to query")
    q.add_argument("--k", type=int, default=8, help="Number of results to return (default: 8)")
    q.add_argument("--json", action="store_true", help="Output raw JSON")

    # ── srag verify ───────────────────────────────────────────────────────────
    v = sub.add_parser("verify", help="Run verification search and detect conflicts")
    v.add_argument("query", type=str, help="Query to verify")
    v.add_argument("--session", type=str, required=True, help="Session name")

    # ── srag sessions ─────────────────────────────────────────────────────────
    sub.add_parser("sessions", help="List all stored sessions")

    # ── srag stale ────────────────────────────────────────────────────────────
    st = sub.add_parser("stale", help="Check if a session is stale")
    st.add_argument("session", type=str, help="Session name to check")
    st.add_argument("--hours", type=int, default=24, help="Max age in hours (default: 24)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # ── Handlers ──────────────────────────────────────────────────────────────

    if args.command == "search":
        from srag.scraper import AnuInfrastructureScraper

        async def _search():
            scraper = AnuInfrastructureScraper(
                max_results=args.results,
                extract_mode="trafilatura"
            )
            results = await scraper.search(args.query)
            if not results:
                print("❌ No results found.")
                return
            for i, r in enumerate(results, 1):
                print(f"\n[{i}] {r['title']}")
                print(f"    URL: {r['url']}")
                print(f"    {r['content'][:200]}...")

        asyncio.run(_search())

    elif args.command == "index":
        from srag import SRag

        async def _index():
            sr = SRag(max_results=args.results)
            result = await sr.search(
                query=args.query,
                session=args.session,
                force_new=args.force_new,
            )
            if result["success"]:
                print(f"\n✅ {result['doc_count']} docs → {result['chunk_count']} chunks indexed into session '{args.session}'")
            else:
                print(f"\n❌ Indexing failed for session '{args.session}'")

        asyncio.run(_index())

    elif args.command == "query":
        from srag import SRag

        sr = SRag()
        results = sr.query(args.query, session=args.session, k=args.k)

        if not results:
            print(f"❌ No results found in session '{args.session}'")
            return

        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            for i, r in enumerate(results, 1):
                print(f"\n[{i}] {r.get('title', 'No Title')}")
                print(f"    URL: {r.get('url', '')}")
                print(f"    {r.get('content', '')[:200]}...")

    elif args.command == "verify":
        from srag import SRag

        async def _verify():
            sr = SRag()
            result = await sr.verify(args.query, session=args.session)

            if result.get("status") == "conflict_detected":
                print(f"\n⚠️  Conflicts detected in '{args.session}':")
                for i, c in enumerate(result["conflicts"], 1):
                    print(f"\n  Conflict {i}:")
                    print(f"    Source A: {c['source_a']['url']}")
                    print(f"    Timestamp: {c['source_a']['timestamp']}")
                    print(f"    Preview: {c['source_a']['content_preview'][:150]}")
                    print(f"    Source B: {c['source_b']['url']}")
                    print(f"    Timestamp: {c['source_b']['timestamp']}")
                    print(f"    Preview: {c['source_b']['content_preview'][:150]}")
                    print(f"    Newest: {c['newest_source']}")
            else:
                print(f"\n✅ No conflicts detected in '{args.session}'")
                print(f"   Docs indexed: {result['doc_count']}")

        asyncio.run(_verify())

    elif args.command == "sessions":
        from srag import SRag

        sr = SRag()
        sessions = sr.list_sessions()
        if not sessions:
            print("No sessions found.")
        else:
            print(f"\n📁 {len(sessions)} session(s) found:\n")
            for s in sessions:
                print(f"  • {s}")

    elif args.command == "stale":
        from srag import SRag

        sr = SRag()
        stale = sr.is_stale(args.session, max_age_hours=args.hours)
        if stale:
            print(f"⚠️  Session '{args.session}' is stale or doesn't exist.")
        else:
            print(f"✅ Session '{args.session}' is fresh (within {args.hours}h).")


if __name__ == "__main__":
    main()

    # TODO: Add logging, error handling, and more robust output formatting. This is a basic scaffold to build upon.