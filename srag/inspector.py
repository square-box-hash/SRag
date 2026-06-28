# srag/inspector.py
"""
Shared data-gathering layer for `srag inspect`.
Used by both cli.py (formats as terminal tables) and server.py (returns JSON).
"""
from __future__ import annotations
import logging
from urllib.parse import urlparse
from typing import Optional

logger = logging.getLogger(__name__)


def get_reputation(db, domain: str = None, topic: str = None) -> dict:
    if "domain_reputation" not in db.table_names():
        return {"available": False, "rows": []}

    rows = db.open_table("domain_reputation").to_pandas().to_dict(orient="records")

    if domain:
        rows = [r for r in rows if domain in r.get("domain", "")]
    if topic:
        rows = [r for r in rows if r.get("topic") == topic]

    rows = sorted(rows, key=lambda r: r.get("retrieval_confidence", 0), reverse=True)

    return {
        "available": True,
        "rows": [
            {
                "domain":               r.get("domain", ""),
                "topic":                r.get("topic", ""),
                "retrieval_confidence": float(r.get("retrieval_confidence", 0)),
                "avg_chunk_quality":    float(r.get("avg_chunk_quality", 0)),
                "useful_hit_rate":      float(r.get("useful_hit_rate", 0)),
                "irrelevance_rate":     float(r.get("irrelevance_rate", 0)),
                "failure_rate":         float(r.get("failure_rate", 0)),
                "avg_latency":          float(r.get("avg_latency", 0)),
                "total_scrapes":        int(r.get("total_scrapes", 0)),
                "last_updated":         r.get("last_updated", ""),
            }
            for r in rows
        ],
    }


def get_lexicon(db, topic: str = None) -> dict:
    if "lexicon" not in db.table_names():
        return {"available": False, "active": [], "candidates": [], "suppressed": []}

    rows = db.open_table("lexicon").to_pandas().to_dict(orient="records")

    if topic:
        rows = [r for r in rows if r.get("topic") == topic]

    def _shape(r):
        return {
            "term":           r.get("term", ""),
            "topic":          r.get("topic", ""),
            "observations":   int(r.get("observations", 0)),
            "avg_confidence": float(r.get("avg_confidence", 0)),
            "status":         r.get("status", ""),
        }

    return {
        "available":  True,
        "active":     [_shape(r) for r in rows if r.get("status") == "active"],
        "candidates": [_shape(r) for r in rows if r.get("status") == "candidate"],
        "suppressed": [_shape(r) for r in rows if r.get("status") == "suppressed"],
    }


def get_chunks(db, session: str) -> dict:
    if session not in db.table_names():
        return {"available": False, "session": session, "rows": [], "by_domain": {}}

    rows = db.open_table(session).to_pandas().to_dict(orient="records")

    by_domain: dict[str, list] = {}
    for r in rows:
        domain = urlparse(r.get("source", "")).netloc.replace("www.", "")
        by_domain.setdefault(domain, []).append(r)

    domain_summary = {
        domain: {
            "chunks": len(chunks),
            "avg_coherence": round(
                sum(float(c.get("coherence_score", 0)) for c in chunks) / len(chunks), 3
            ),
        }
        for domain, chunks in by_domain.items()
    }

    return {
        "available":      True,
        "session":        session,
        "total_chunks":   len(rows),
        "total_domains":  len(by_domain),
        "by_domain":      domain_summary,
        "rows":           [
            {
                "content":        r.get("content", ""),
                "source":         r.get("source", ""),
                "coherence_score": float(r.get("coherence_score", 0)),
                "timestamp":      r.get("timestamp", ""),
            }
            for r in rows
        ],
    }


def get_domain_filters(db) -> dict:
    if "domain_filters" not in db.table_names():
        return {"available": False, "rows": []}

    rows = db.open_table("domain_filters").to_pandas().to_dict(orient="records")

    return {
        "available": True,
        "rows": [
            {
                "domain":               r.get("domain", ""),
                "selectors":            list(r.get("selectors") or []),
                "source":               r.get("source", ""),
                "low_coherence_streak": int(r.get("low_coherence_streak", 0)),
                "total_scrapes":        int(r.get("total_scrapes", 0)),
                "flagged":              bool(r.get("flagged", False)),
                "last_updated":         r.get("last_updated", ""),
            }
            for r in sorted(rows, key=lambda r: r.get("flagged", False), reverse=True)
        ],
    }


def inspect(
    db,
    show:    str            = "all",
    session: Optional[str]  = None,
    domain:  Optional[str]  = None,
    topic:   Optional[str]  = None,
) -> dict:
    """
    Single entry point for both cli.py and server.py.
    Returns a plain dict — CLI formats it as terminal tables,
    server returns it as JSON.
    """
    result = {}

    if show in ("all", "reputation"):
        result["reputation"] = get_reputation(db, domain=domain, topic=topic)

    if show in ("all", "lexicon", "candidates"):
        result["lexicon"] = get_lexicon(db, topic=topic)

    if show in ("all", "chunks") and session:
        result["chunks"] = get_chunks(db, session=session)

    if show in ("all", "domain-filters"):
        result["domain_filters"] = get_domain_filters(db)

    return result