import argparse
import asyncio
import json
import logging
import re
from urllib.parse import urlparse
from dotenv import load_dotenv
from lancedb import db

load_dotenv()


def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(format="%(levelname)s [%(name)s] %(message)s", level=level)
    if not debug:
        for name in [
            "sentence_transformers", "huggingface_hub", "httpx",
            "playwright", "trafilatura", "trafilatura.core",
            "trafilatura.utils", "srag.scraper",
        ]:
            logging.getLogger(name).setLevel(logging.ERROR)
            logging.getLogger(name).propagate = False


# ── Content detection ─────────────────────────────────────────────────────────

def _is_garbage(text: str) -> bool:
    """Detect binary garbage — high ratio of non-printable chars."""
    if not text:
        return True
    non_printable = sum(1 for c in text if ord(c) > 127 and ord(c) < 160)
    return (non_printable / max(len(text), 1)) > 0.05


def _is_nav_heavy(text: str) -> bool:
    """Detect navigation/menu fragments with little actual content."""
    lines       = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return True
    short_lines = sum(1 for l in lines if len(l) < 30)
    return (short_lines / max(len(lines), 1)) > 0.70


def _detect_format(text: str) -> str:
    """
    Detect content format.
    Returns: 'table' | 'paragraph' | 'nav' | 'garbage'
    """
    if _is_garbage(text):
        return "garbage"
    if "|" in text and text.count("|") > 4:
        return "table"
    if _is_nav_heavy(text):
        return "nav"
    return "paragraph"


# ── Table renderer ────────────────────────────────────────────────────────────

def _render_table(text: str, max_rows: int = 15) -> str:
    """
    Parse markdown-style or pipe-delimited tables and render as aligned columns.
    Falls back to cleaned text if parsing fails.
    """
    lines  = [l.strip() for l in text.splitlines() if l.strip()]
    rows   = []

    for line in lines:
        if "|" in line:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            # Skip separator lines like |---|---|
            if all(set(c) <= set("-: ") for c in cells):
                continue
            if cells:
                rows.append(cells)

    if not rows:
        return text[:600]

    # Limit rows
    rows = rows[:max_rows]

    # Compute column widths
    num_cols = max(len(r) for r in rows)
    widths   = [0] * num_cols
    for row in rows:
        for i, cell in enumerate(row):
            if i < num_cols:
                widths[i] = max(widths[i], min(len(cell), 24))

    # Render
    lines_out = []
    for idx, row in enumerate(rows):
        cells = []
        for i in range(num_cols):
            cell = row[i] if i < len(row) else ""
            cell = cell[:24]
            cells.append(f"{cell:<{widths[i]}}")
        lines_out.append("  " + "  ".join(cells))
        if idx == 0:
            lines_out.append("  " + "  ".join("─" * w for w in widths))

    return "\n".join(lines_out)


# ── Paragraph renderer ────────────────────────────────────────────────────────

def _render_paragraphs(text: str, max_paras: int = 3, max_chars: int = 300) -> tuple[str, int]:
    """
    Render top N paragraphs from content.
    Returns (rendered_text, remaining_paragraph_count).
    """
    # Split on double newlines or sentence clusters
    raw_paras = re.split(r"\n{2,}", text)
    paras     = [
        p.strip() for p in raw_paras
        if len(p.strip()) > 60
        and not _is_nav_heavy(p)
        and not _is_garbage(p)
    ]

    if not paras:
        # Fallback — split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        paras     = [" ".join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
        paras     = [p for p in paras if len(p.strip()) > 60]

    remaining = max(0, len(paras) - max_paras)
    selected  = paras[:max_paras]

    rendered = []
    for para in selected:
        if len(para) > max_chars:
            # Trim at sentence boundary
            trimmed = para[:max_chars]
            last    = max(trimmed.rfind(". "), trimmed.rfind("? "), trimmed.rfind("! "))
            if last > max_chars * 0.6:
                trimmed = trimmed[:last + 1]
            rendered.append(trimmed.strip() + "...")
        else:
            rendered.append(para.strip())

    return "\n\n  ".join(rendered), remaining


# ── Topic badge ───────────────────────────────────────────────────────────────

def _topic_badge(topic: str) -> str:
    badges = {
        "tech":          "⚙",
        "finance":       "₹",
        "sports":        "🏁",
        "science":       "🔬",
        "news":          "📰",
        "education":     "📚",
        "entertainment": "🎬",
        "health":        "🏥",
        "general":       "◆",
    }
    return badges.get(topic, "◆")


# ── Main formatters ───────────────────────────────────────────────────────────

def _print_search_results(
    results: list[dict],
    query:   str   = "",
    debug:   bool  = False,
) -> None:
    """
    Google AI overview-style search output.
    Tables rendered as columns, paragraphs as clean text,
    nav/garbage filtered and counted.
    """
    if not results:
        print("\n  ❌ No results found.")
        return

    width = 52

    # Header
    print(f"\n  {'═' * width}")
    q_display = query[:40] + "..." if len(query) > 40 else query
    print(f"  {q_display:<{width}}")
    print(f"  {'═' * width}\n")

    shown    = 0
    filtered = 0

    for r in results:
        content = r.get("content", "").strip()
        domain  = r.get("domain", _get_domain(r.get("url", "")))
        title   = r.get("title", "").strip()
        topic   = r.get("topic", "general")
        date    = r.get("published_date", "")[:10] if r.get("published_date") else ""
        fmt     = _detect_format(content)

        if fmt in ("garbage", "nav"):
            filtered += 1
            if debug:
                print(f"  [debug] filtered {domain} — {fmt}")
            continue

        shown += 1
        badge    = _topic_badge(topic)
        fmt_tag  = "table" if fmt == "table" else "article"
        meta     = f"{badge} {topic} • {fmt_tag}" + (f" • {date}" if date else "")

        # Source header
        print(f"  [{shown}] {domain:<28} {meta}")
        print(f"  {'─' * width}")

        # Title if meaningful
        if title and domain.lower() not in title.lower():
            short_title = title[:60] + "..." if len(title) > 60 else title
            print(f"  {short_title}")
            print()

        # Content
        if fmt == "table":
            rendered = _render_table(content)
            print(rendered)
        else:
            rendered, remaining = _render_paragraphs(content)
            print(f"  {rendered}")
            if remaining > 0:
                print(f"\n  [{remaining} more paragraph{'s' if remaining > 1 else ''} not shown]")

        if debug:
            score     = r.get("score", 0)
            coherence = r.get("coherence_score", r.get("coherence", 0))
            tokens    = len(content) // 4
            print(f"\n  [debug] score={score:.2f} coherence={coherence:.3f} tokens={tokens}")

        print(f"\n  {'─' * width}\n")

    # Footer
    print(f"  {'═' * width}")
    parts = [f"{shown} source{'s' if shown != 1 else ''}"]
    if filtered:
        parts.append(f"{filtered} filtered (nav/garbage)")
    print(f"  {'  •  '.join(parts)}")
    print(f"  {'═' * width}\n")


def _print_query_results(
    results: list[dict],
    query:   str  = "",
    debug:   bool = False,
) -> None:
    """
    Hybrid table + paragraph format for srag query.
    Summary ranking table at top, full content below.
    """
    if not results:
        print("\n  ❌ No results found.")
        return

    width = 52

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n  {'═' * width}")
    if query:
        q_display = query[:40] + "..." if len(query) > 40 else query
        print(f"  {q_display}")
        print(f"  {'─' * width}")

    # Header row
    print(f"  {'#':<4} {'Source':<28} {'Score':>6}  {'Tokens':>6}")
    print(f"  {'─' * 4} {'─' * 28} {'─' * 6}  {'─' * 6}")

    for i, r in enumerate(results, 1):
        domain  = _get_domain(r.get("url", r.get("source", "")))[:26]
        score   = r.get("score", r.get("_distance", 0))
        tokens  = len(r.get("content", "")) // 4

        if score > 1.0:
            score = max(0.0, 1.0 - score)

        print(f"  {i:<4} {domain:<28} {score:>6.2f}  {tokens:>6}")

    print(f"  {'═' * width}\n")

    # ── Full content per result ───────────────────────────────────────────────
    for i, r in enumerate(results, 1):
        domain  = _get_domain(r.get("url", r.get("source", "")))
        title   = r.get("title", "").strip()
        content = r.get("content", "").strip()
        url     = r.get("url", r.get("source", ""))
        fmt     = _detect_format(content)

        print(f"  [{i}] {domain}")
        print(f"  {'─' * width}")

        if title and domain.lower() not in title.lower():
            short_title = title[:70] + "..." if len(title) > 70 else title
            print(f"  {short_title}")
            print()

        if fmt == "table":
            print(_render_table(content))
        elif fmt == "garbage" or fmt == "nav":
            print("  [content filtered — low quality]")
        else:
            rendered, remaining = _render_paragraphs(content, max_paras=2)
            print(f"  {rendered}")
            if remaining > 0:
                print(f"\n  [{remaining} more paragraph{'s' if remaining > 1 else ''} not shown]")

        if debug:
            chunk_idx = r.get("chunk_index", "?")
            coherence = r.get("coherence_score", 0)
            print(f"\n  [debug] chunk={chunk_idx} coherence={coherence:.3f} url={url}")

        print(f"\n  {'─' * width}\n")


def _print_index_result(result: dict, session: str, debug: bool = False) -> None:
    """Clean structured output for srag index."""
    width = 52
    print(f"\n  {'═' * width}")

    if result["success"]:
        print(f"  ✅ Indexed into '{session}'")
        print(f"  {'─' * width}")
        print(f"  {'Documents scraped':<28} {result.get('doc_count', 0):>6}")
        print(f"  {'Chunks created':<28} {result.get('chunk_count', 0):>6}")
        print(f"  {'Chunks indexed':<28} {result.get('indexed_count', 0):>6}")
        print(f"  {'Topic detected':<28} {result.get('topic', '?'):>6}")
    else:
        reason = result.get("reason", "unknown")
        print(f"  ❌ Indexing failed — {reason}")
        print(f"  {'─' * width}")
        print(f"  {'Documents found':<28} {result.get('doc_count', 0):>6}")

    if debug and "debug" in result:
        d = result["debug"]
        print(f"  {'─' * width}")
        print(f"  [debug] quality:     {d['quality']}")
        print(f"  [debug] concurrency: global={d['concurrency']['global_concurrency']} p95={d['concurrency']['p95_latency']}s")
        domains = list(d["collector"]["domains_data"].keys())
        print(f"  [debug] domains:     {', '.join(domains[:5])}")

    print(f"  {'═' * width}\n")


def _print_verify_result(result: dict, session: str) -> None:
    """Structured conflict report for srag verify."""
    width = 52
    print(f"\n  {'═' * width}")

    if result.get("status") == "conflict_detected":
        conflicts = result.get("conflicts", [])
        print(f"  ⚠️  {len(conflicts)} conflict{'s' if len(conflicts) != 1 else ''} detected in '{session}'")
        print(f"  {'─' * width}")
        for i, c in enumerate(conflicts[:5], 1):  # show max 5
            print(f"\n  Conflict {i}:")
            print(f"    Source A:  {c['source_a']['url'][:50]}")
            print(f"    Dated:     {c['source_a']['timestamp'][:10]}")
            preview_a = c['source_a']['content_preview'][:120].replace('\n', ' ')
            print(f"    Preview:   {preview_a}...")
            print()
            print(f"    Source B:  {c['source_b']['url'][:50]}")
            print(f"    Dated:     {c['source_b']['timestamp'][:10]}")
            preview_b = c['source_b']['content_preview'][:120].replace('\n', ' ')
            print(f"    Preview:   {preview_b}...")
            print(f"    Newest:    {c['newest_source'][:50]}")
        if len(conflicts) > 5:
            print(f"\n  [{len(conflicts) - 5} more conflicts not shown]")
    else:
        print(f"  ✅ No conflicts in '{session}'")
        print(f"  {'─' * width}")
        print(f"  {'Documents indexed':<28} {result.get('doc_count', 0):>6}")

    print(f"\n  {'═' * width}\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "unknown"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog        = "srag",
        description = "SRag — Smart RAG CLI",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  srag search "F1 2026 championship standings"
  srag search "GST rate India" --debug
  srag index "python asyncio tutorial" --session python_async
  srag index "GST rate India" --session gst --results 15 --force-new
  srag query "what is the rate" --session gst
  srag query "what is the rate" --session gst --k 10 --debug
  srag verify "GST rate India" --session gst_verify
  srag sessions
  srag stale gst --hours 48
        """
    )

    parser.add_argument(
        "--debug", "-d",
        action  = "store_true",
        default = False,
        help    = "Enable debug output",
    )

    sub = parser.add_subparsers(dest="command", metavar="command")

    s = sub.add_parser("search", help="Search and display results")
    s.add_argument("query",     type=str)
    s.add_argument("--results", type=int, default=12)

    idx = sub.add_parser("index", help="Search, scrape and index into a session")
    idx.add_argument("query",       type=str)
    idx.add_argument("--session",   type=str, required=True)
    idx.add_argument("--results",   type=int, default=12)
    idx.add_argument("--force-new", action="store_true")

    q = sub.add_parser("query", help="Semantic search within a stored session")
    q.add_argument("query",     type=str)
    q.add_argument("--session", type=str, required=True)
    q.add_argument("--k",       type=int, default=8)
    q.add_argument("--json",    action="store_true")

    rd = sub.add_parser("read", help="Fetch and read a URL directly in terminal")
    rd.add_argument("url",       type=str)
    rd.add_argument("--width",   type=int, default=72, help="Line wrap width (default: 72)")
    rd.add_argument("--raw",     action="store_true",  help="Show raw text without formatting")

    # in the subparser section
    ins = sub.add_parser("inspect", help="Inspect reputation, chunks and signals for a query/session")
    ins.add_argument("query",       type=str,            nargs="?", default=None)
    ins.add_argument("--session",   type=str,            default=None)
    ins.add_argument("--domain",    type=str,            default=None)
    ins.add_argument("--topic",     type=str,            default=None)
    ins.add_argument("--show",      type=str,            default="all",
                    choices=["all", "reputation", "lexicon", "chunks", "collector","candidates"])
    ins.add_argument("--verbose", "-v", action="store_true", default=False,
                 help="Show all fields for matched domains")

    v = sub.add_parser("verify", help="Run verification search and detect conflicts")
    v.add_argument("query",     type=str)
    v.add_argument("--session", type=str, required=True)

    sub.add_parser("sessions", help="List all stored sessions")

    st = sub.add_parser("stale", help="Check if a session is stale")
    st.add_argument("session", type=str)
    st.add_argument("--hours", type=int, default=24)

    args = parser.parse_args()
    _setup_logging(args.debug)

    if not args.command:
        parser.print_help()
        return

    if args.command == "search":
        from srag.scraper import AnuInfrastructureScraper
        from srag.chunker import SmartChunker
        from srag.topic_classifier import TopicClassifier
        from sentence_transformers import SentenceTransformer

        async def _search():
            _model    = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            scraper   = AnuInfrastructureScraper(max_results=args.results)
            chunker   = SmartChunker(model=_model, max_tokens=256)
            tc        = TopicClassifier()
            topic     = tc.predict(args.query).primary

            docs   = await scraper.get_facts(args.query)
            chunks = chunker.chunk_docs(docs)

            # Best chunk per domain
            by_domain: dict[str, dict] = {}
            for chunk in chunks:
                domain = _get_domain(chunk.get("source", ""))
                if domain not in by_domain:
                    by_domain[domain] = chunk

            display = [
                {
                    "domain":         domain,
                    "url":            c.get("source", ""),
                    "title":          c.get("title", ""),
                    "content":        c.get("content", ""),
                    "score":          c.get("coherence_score", 0.0),
                    "coherence_score": c.get("coherence_score", 0.0),
                    "published_date": c.get("timestamp", ""),
                    "topic":          topic,
                }
                for domain, c in by_domain.items()
            ]

            if args.debug:
                print(f"\n  [debug] {len(docs)} docs → {len(chunks)} chunks → {len(display)} domains")
                print(f"  [debug] topic={topic}")

            _print_search_results(display, query=args.query, debug=args.debug)

            # ── Cache for `srag read <index>` ─────────────────────────
            import json, tempfile, os
            LAST_SEARCH_CACHE = os.path.join(tempfile.gettempdir(), "srag_last_search.json")
            with open(LAST_SEARCH_CACHE, "w") as f:
                json.dump(display, f)

        asyncio.run(_search())

    elif args.command == "index":
        from srag import SRag

        async def _index():
            sr     = SRag(max_results=args.results)
            result = await sr.search(
                query     = args.query,
                session   = args.session,
                force_new = args.force_new,
                debug     = args.debug,
            )
            _print_index_result(result, args.session, debug=args.debug)

        asyncio.run(_index())

    elif args.command == "query":
        from srag import SRag

        sr      = SRag()
        results = sr.query(args.query, session=args.session, k=args.k, debug=args.debug)

        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            _print_query_results(results, query=args.query, debug=args.debug)

    elif args.command == "verify":
        from srag import SRag

        async def _verify():
            sr     = SRag()
            result = await sr.verify(args.query, session=args.session, debug=args.debug)
            _print_verify_result(result, args.session)

        asyncio.run(_verify())

    elif args.command == "sessions":
        from srag import SRag

        sr       = SRag()
        sessions = sr.list_sessions()
        width    = 52

        print(f"\n  {'═' * width}")
        print(f"  Sessions ({len(sessions)} total)")
        print(f"  {'─' * width}")

        if not sessions:
            print("  No sessions found.")
        else:
            for s in sessions:
                stale = sr.is_stale(s)
                tag   = "⚠️  stale" if stale else "✅ fresh"
                print(f"  {s:<36} {tag}")

        print(f"  {'═' * width}\n")

    elif args.command == "read":
        from srag.scraper import AnuInfrastructureScraper
        from srag.chunker import SmartChunker
        from sentence_transformers import SentenceTransformer
        import textwrap

        async def _read():
            # ── Resolve URL from arg or cached search index ───────────
            import json, os, tempfile
            LAST_SEARCH_CACHE = os.path.join(tempfile.gettempdir(), "srag_last_search.json")

            if args.url.isdigit():
                try:
                    with open(LAST_SEARCH_CACHE) as f:
                        cache = json.load(f)
                    idx = int(args.url) - 1   # 1-indexed like your search output
                    if idx >= len(cache):
                        print(f"  ❌ Only {len(cache)} results cached.")
                        return
                    url = cache[idx]["url"]
                    print(f"  → Reading result #{args.url}: {cache[idx]['title'][:60]}...")
                except FileNotFoundError:
                    print("  ❌ No cached search found. Run `srag search` first.")
                    return
            else:
                url = args.url

            scraper = AnuInfrastructureScraper(max_results=1)
            _model  = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            chunker = SmartChunker(model=_model, max_tokens=256)
            width   = args.width

            print(f"\n  Fetching {url}...\n")

            # Fetch single URL directly
            import httpx
            import random
            from srag.scraper import REALISTIC_HEADERS

            async with httpx.AsyncClient(
                timeout = 15.0,
                follow_redirects = True
            ) as client:
                doc = await scraper._fetch_and_clean(
                    client,
                    url,
                    "Article",
                )

            if not doc:
                print("  ❌ Failed to fetch or extract content.")
                return

            chunks = chunker.chunk(
                text       = doc.get("content", ""),
                source_url = url,
                title      = doc.get("title", ""),
            )

            if not chunks:
                print("  ❌ No readable content found.")
                return

            # ── Render ────────────────────────────────────────────────────────
            title     = doc.get("title", "").strip()
            author    = doc.get("author", "")
            timestamp = doc.get("timestamp", "")[:10]

            print(f"  {'═' * width}")
            if title:
                 # Word wrap title
                 wrapped_title = textwrap.fill(title, width=width - 2)
                 for line in wrapped_title.splitlines():
                     print(f"  {line}")
            print(f"  {'─' * width}")

            meta_parts = []
            if author:
                meta_parts.append(f"By {author}")
            if timestamp:
                meta_parts.append(f"Published on {timestamp}")
            meta_parts.append(_get_domain(url))
            print(f"  {' • '.join(meta_parts)}")
            print(f"  {'═' * width}\n")

            if args.raw:
                # Raw mode — just print full text
                print(textwrap.fill(doc.get("content", ""), width=width))
            else:
                # Render chunks with format detection
                seen = set()
                for chunk in chunks:
                    content = chunk.get("content", "").strip()
                    coherence = chunk.get("coherence_score", 0)

                    if coherence < 0.25: 
                        continue

                    fmt = _detect_format(content)

                    if fmt == "table":
                        print(_render_table(content))
                    elif fmt in ("garbage", "nav"):
                        continue
                    else:
                        paragraphs = [
                            p.strip() for p in content.split("\n\n")
                            if len(p.strip()) > 40
                        ]
                        # Fallback: if no paragraphs found after split, treat whole chunk as one
                        if not paragraphs and len(content.strip()) > 40:
                            paragraphs = [content.strip()]
                        for p in paragraphs:
                            # Normalize whitespace (handles articles with \r\n or collapsed lines)
                            p = " ".join(p.split())
                            if p in seen:
                                continue
                            seen.add(p)
                            wrapped = textwrap.fill(p, width=width)
                            for line in wrapped.splitlines():
                                print(f"  {line}")
                            print()

            print(f"  {'─' * width}")
            print(f"  {url}")
            print(f"  {'═' * width}\n")

        asyncio.run(_read())

    elif args.command == "inspect":
        from srag import SRag
        import lancedb

        sr    = SRag()
        db    = sr._orchestrator.indexer.db
        width = 60

        print(f"\n  {'═' * width}")
        print(f"  SRag Inspector")
        print(f"  {'═' * width}")

        # ── Reputation table ──────────────────────────────────────────────────
        if args.show in ("all", "reputation"):
            print(f"\n  DOMAIN REPUTATION")
            print(f"  {'─' * width}")
            
            if "domain_reputation" not in db.table_names():
                print("  No reputation data yet — run some searches first.")
            else:
                table = db.open_table("domain_reputation")
                rows  = table.to_pandas().to_dict(orient="records")
                
                # Filter by domain or topic if specified
                if args.domain:
                    rows = [r for r in rows if args.domain in r.get("domain", "")]
                if args.topic:
                    rows = [r for r in rows if r.get("topic") == args.topic]
                    
                rows = sorted(rows, key=lambda r: r.get("retrieval_confidence", 0), reverse=True)
                
                if not rows:
                    print("  No matching records.")
                    
                else:
                    print(f"  {'Domain':<28} {'Topic':<12} {'Conf':>5}  {'Quality':>7}  {'Hit%':>5}  {'Scrapes':>7}")
                    print(f"  {'─'*28} {'─'*12} {'─'*5}  {'─'*7}  {'─'*5}  {'─'*7}")
                    
                    for r in rows[:20]:
                        domain   = r.get("domain", "")[:26]
                        topic    = r.get("topic", "")[:10]
                        conf     = float(r.get("retrieval_confidence", 0))
                        quality  = float(r.get("avg_chunk_quality", 0))
                        hit_rate = float(r.get("useful_hit_rate", 0))
                        scrapes  = int(r.get("total_scrapes", 0))
                        
                        if conf >= 0.75:
                            indicator = "🟢"
                        elif conf >= 0.55:
                            indicator = "🟡"
                        else:
                            indicator = "🔴"
                            
                        print(f"  {indicator} {domain:<26} {topic:<12} {conf:>5.2f}  {quality:>7.2f}  {hit_rate:>5.0%}  {scrapes:>7}")
                        
                    if len(rows) > 20:
                        print(f"\n  [{len(rows) - 20} more domains not shown — use --domain to filter]")

                    if args.verbose and rows:
                        print(f"\n  {r.get('domain')} / {r.get('topic')}")
                        print(f"  {'─' * 40}")
                        print(f"  retrieval_confidence : {r.get('retrieval_confidence', 0):.4f}")
                        print(f"  avg_chunk_quality    : {r.get('avg_chunk_quality', 0):.4f}")
                        print(f"  useful_hit_rate      : {r.get('useful_hit_rate', 0):.4f}")
                        print(f"  irrelevance_rate     : {r.get('irrelevance_rate', 0):.4f}")
                        print(f"  failure_rate         : {r.get('failure_rate', 0):.4f}")
                        print(f"  avg_latency          : {r.get('avg_latency', 0):.3f}s")
                        print(f"  total_scrapes        : {r.get('total_scrapes', 0)}")
                        print(f"  last_updated         : {r.get('last_updated', '')[:19]}")


        # ── Lexicon table ─────────────────────────────────────────────────────
        if args.show in ("all", "lexicon","candidates"):
            print(f"\n  LEXICON")
            print(f"  {'─' * width}")
            
            if "lexicon" not in db.table_names():
                print("  No lexicon data yet — run some searches first.")
            else:
                table = db.open_table("lexicon")
                rows  = table.to_pandas().to_dict(orient="records")
                
                if args.topic:
                    rows = [r for r in rows if r.get("topic") == args.topic]
                    
                active = [r for r in rows if r.get("status") == "active"]
                candidates = [r for r in rows if r.get("status") == "candidate"]
                suppressed = [r for r in rows if r.get("status") == "suppressed"]
                
                print(f"  Active: {len(active)}  Candidate: {len(candidates)}  Suppressed: {len(suppressed)}")
                print()

                if active:
                    print(f"  {'Term':<24} {'Topic':<12} {'Obs':>5}  {'AvgConf':>7}  {'Status':<10}")
                    print(f"  {'─'*24} {'─'*12} {'─'*5}  {'─'*7}  {'─'*10}")
                    for r in sorted(active, key=lambda x: x.get("avg_confidence", 0), reverse=True)[:15]:
                        print(
                            f"  {'✅'} {r.get('term',''):<22} {r.get('topic',''):<12} "
                            f"{r.get('observations',0):>5}  {r.get('avg_confidence',0):>7.3f}  "
                            f"{r.get('status',''):<10}"
                        )

                else:
                    print("No active terms yet — terms need 8+ high-confidence observations to graduate.")

                if args.show in ("candidates", "all") or not active:
                    if candidates:
                        print(f"\n  {'Term':<24} {'Topic':<12} {'Obs':>5}  {'AvgConf':>7}")
                        print(f"  {'─'*24} {'─'*12} {'─'*5}  {'─'*7}")
                        for r in sorted(candidates, key=lambda x: x.get("observations", 0), reverse=True)[:20]:
                            print(
                                f"  {'🔵'} {r.get('term',''):<22} {r.get('topic',''):<12} "
                                f"{r.get('observations',0):>5}  {r.get('avg_confidence',0):>7.3f}"
                            )
                    else:
                        print("  No candidates yet.")


        # --─ Chunks and collector tables could be added here similarly if needed
        if args.show in ("all", "chunks") and args.session:
            print(f"\n  CHUNKS IN SESSION '{args.session}'")
            print(f"  {'─' * width}")
            
            if "chunks" not in db.table_names():
                print(f"  Session '{args.session}' not found or no chunks indexed yet.")
            else:
                table = db.open_table("chunks")
                rows  = table.to_pandas().to_dict(orient="records")
                session_chunks = [r for r in rows if r.get("session") == args.session]
                
                #Group by domains
                by_domain: dict[str, list] = {}
                for r in rows:
                    from urllib.parse import urlparse
                    domain = urlparse(r.get("url", "")).netloc.replace("www.", "")
                    by_domain.setdefault(domain, []).append(r)

                print(f"  Total chunks: {len(rows)}  Domains: {len(by_domain)}")
                print()
                print(f"  {'Domain':<28} {'Chunks':>6}  {'AvgCoherence':>12}")
                print(f"  {'─'*28} {'─'*6}  {'─'*12}")

                for domain, chunks in sorted(by_domain.items(), key=lambda x: len(x[1]), reverse=True):
                    avg_coh = sum(float(c.get("coherence_score", 0)) for c in chunks) / len(chunks)
                    print(f"  {domain:<28} {len(chunks):>6}  {avg_coh:>12.3f}")

                # If query provided, show top matching chunks
                if args.query:
                    print(f"\n  TOP CHUNKS FOR '{args.query}'")
                    print(f"  {'─' * width}")
                    results = sr.query(args.query, session=args.session, k=5)
                    for i, r in enumerate(results, 1):
                        url     = r.get("url", r.get("source", ""))[:50]
                        domain  = urlparse(r.get("url", r.get("source", ""))).netloc.replace("www.", "")
                        coh     = r.get("score", 0)
                        content = r.get("content", "").strip()
                        fmt     = _detect_format(content)

                        print(f"  [{i}] {domain:<28} (coherence={coh:.3f})")
                        print(f"  {'─' * width}")
                        if fmt == "table":
                            print(_render_table(content))
                        elif fmt in ("garbage", "nav"):
                            print("  [content filtered — low quality]")
                        else:
                            rendered, remaining = _render_paragraphs(content, max_paras=2)
                            print(f"  {rendered}")
                            if remaining > 0:
                                print(f"\n  [{remaining} more paragraph{'s' if remaining > 1 else ''} not shown]")

        print(f"\n  {'═' * width}\n")

    elif args.command == "stale":
        from srag import SRag

        sr    = SRag()
        stale = sr.is_stale(args.session, max_age_hours=args.hours)
        width = 52

        print(f"\n  {'═' * width}")
        if stale:
            print(f"  ⚠️  '{args.session}' is stale or doesn't exist")
        else:
            print(f"  ✅ '{args.session}' is fresh (within {args.hours}h)")
        print(f"  {'═' * width}\n")


if __name__ == "__main__":
    main()