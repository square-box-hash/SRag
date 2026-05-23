import os, json, csv
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# ── Base ──────────────────────────────────────────────────────────────────────

def _make_doc(content: str, source: str, title: str = "", timestamp: str = "") -> dict:
    return {
        "content":   content,
        "source":    source,
        "title":     title or Path(source).name,
        "timestamp": timestamp or datetime.now().isoformat(),
        "author":    "",
        "image":     "",
    }


# ── PDF ───────────────────────────────────────────────────────────────────────

def ingest_pdf(path: str, max_pages: int = 500) -> List[dict]:
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ImportError("pymupdf required: pip install pymupdf")

    doc   = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        text = page.get_text().strip()
        if text:
            pages.append(text)
    doc.close()

    if not pages:
        return []

    # One doc per page for better chunking granularity
    return [
        _make_doc(
            content = text,
            source  = f"file://{os.path.abspath(path)}#page{i+1}",
            title   = f"{Path(path).stem} — Page {i+1}",
        )
        for i, text in enumerate(pages)
    ]


# ── DOCX ──────────────────────────────────────────────────────────────────────

def ingest_docx(path: str) -> List[dict]:
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx required: pip install python-docx")

    doc        = Document(path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    content    = "\n\n".join(paragraphs)

    if not content:
        return []

    return [_make_doc(
        content = content,
        source  = f"file://{os.path.abspath(path)}",
        title   = Path(path).stem,
    )]


# ── TXT ───────────────────────────────────────────────────────────────────────

def ingest_txt(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip()

    if not content:
        return []

    return [_make_doc(
        content = content,
        source  = f"file://{os.path.abspath(path)}",
        title   = Path(path).stem,
    )]


# ── CSV ───────────────────────────────────────────────────────────────────────

def ingest_csv(path: str, max_rows: int = 10000) -> List[dict]:
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required: pip install pandas")

    df   = pd.read_csv(path, nrows=max_rows)
    docs = []

    for i, row in df.iterrows():
        content = " | ".join([f"{col}: {val}" for col, val in row.items() if str(val).strip()])
        if content:
            docs.append(_make_doc(
                content = content,
                source  = f"file://{os.path.abspath(path)}#row{i+1}",
                title   = f"{Path(path).stem} — Row {i+1}",
            ))

    return docs


# ── JSON ──────────────────────────────────────────────────────────────────────

def ingest_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []

    def _flatten(obj, prefix="") -> str:
        if isinstance(obj, dict):
            return " | ".join([
                f"{prefix}{k}: {_flatten(v, '')}"
                for k, v in obj.items()
            ])
        elif isinstance(obj, list):
            return " ".join([_flatten(item) for item in obj])
        else:
            return str(obj)

    if isinstance(data, list):
        for i, item in enumerate(data):
            content = _flatten(item)
            if content.strip():
                docs.append(_make_doc(
                    content = content,
                    source  = f"file://{os.path.abspath(path)}#item{i}",
                    title   = f"{Path(path).stem} — Item {i}",
                ))
    else:
        content = _flatten(data)
        if content.strip():
            docs.append(_make_doc(
                content = content,
                source  = f"file://{os.path.abspath(path)}",
                title   = Path(path).stem,
            ))

    return docs


# ── SQLite ────────────────────────────────────────────────────────────────────

def ingest_sqlite(
    db_path:   str,
    table:     Optional[str] = None,
    query:     Optional[str] = None,
    max_rows:  int = 10000,
) -> List[dict]:
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()

    if query:
        cur.execute(query)
    elif table:
        cur.execute(f"SELECT * FROM {table} LIMIT {max_rows}")
    else:
        raise ValueError("Provide either table or query for SQLite ingestion")

    rows = cur.fetchall()
    conn.close()

    docs = []
    for i, row in enumerate(rows):
        content = " | ".join([f"{k}: {row[k]}" for k in row.keys() if row[k]])
        if content:
            docs.append(_make_doc(
                content = content,
                source  = f"sqlite://{os.path.abspath(db_path)}#{table or 'query'}:row{i}",
                title   = f"{Path(db_path).stem} — {table or 'query'} Row {i}",
            ))

    return docs


# ── PostgreSQL ────────────────────────────────────────────────────────────────

def ingest_postgres(
    connection_string: str,
    query:             str,
    max_rows:          int = 10000,
) -> List[dict]:
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        raise ImportError("psycopg2 required: pip install psycopg2-binary")

    conn = psycopg2.connect(connection_string)
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(query)
    rows = cur.fetchmany(max_rows)
    conn.close()

    docs = []
    for i, row in enumerate(rows):
        content = " | ".join([f"{k}: {v}" for k, v in row.items() if v])
        if content:
            docs.append(_make_doc(
                content = content,
                source  = f"postgresql://query#row{i}",
                title   = f"PostgreSQL Query — Row {i}",
            ))

    return docs


# ── Router ────────────────────────────────────────────────────────────────────

class DocumentIngestor:
    """
    Single entry point for all local file and database ingestion.
    Returns standard doc dicts compatible with SmartChunker.
    """

    def __init__(self, config=None):
        self.config = config

    def ingest(self, source: str, **kwargs) -> List[dict]:
        """
        Auto-detect source type and ingest.
        source can be: file path, folder path, sqlite:// or postgresql:// URI
        """
        if source.startswith("sqlite://"):
            path = source.replace("sqlite://", "")
            return ingest_sqlite(path, **kwargs)

        if source.startswith("postgresql://") or source.startswith("postgres://"):
            return ingest_postgres(source, **kwargs)

        path = Path(source)

        if path.is_dir():
            return self.ingest_folder(str(path), **kwargs)

        ext = path.suffix.lower()
        if ext == ".pdf":
            max_pages = self.config.ingestor_max_pdf_pages if self.config else 500
            return ingest_pdf(str(path), max_pages=max_pages)
        elif ext == ".docx":
            return ingest_docx(str(path))
        elif ext == ".txt":
            return ingest_txt(str(path))
        elif ext == ".csv":
            max_rows = self.config.ingestor_csv_max_rows if self.config else 10000
            return ingest_csv(str(path), max_rows=max_rows)
        elif ext == ".json":
            return ingest_json(str(path))
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def ingest_folder(self, folder: str, **kwargs) -> List[dict]:
        """Ingest all supported files in a folder."""
        supported = {".pdf", ".docx", ".txt", ".csv", ".json"}
        docs = []
        for f in Path(folder).rglob("*"):
            if f.suffix.lower() in supported:
                try:
                    docs.extend(self.ingest(str(f), **kwargs))
                    print(f"  ✅ Ingested: {f.name}")
                except Exception as e:
                    print(f"  ❌ Failed: {f.name} — {e}")
        return docs