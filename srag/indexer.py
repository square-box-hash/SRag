import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import lancedb
import pyarrow as pa
from dotenv import load_dotenv

from srag.embeddings import (
    DEFAULT_LOCAL_MODEL,
    Embedder,
    EmbeddingMismatchError,
    EmbeddingSpec,
    LocalEmbedder,
)

load_dotenv()

META_FILENAME = "index_meta.json"


@dataclass
class IndexedChunk:
    id: str
    url: str
    title: str
    content: str
    timestamp: str
    author: Optional[str]
    chunk_index: int
    coherence_score: float


class SRagIndexer:
    """LanceDB-backed chunk store whose tables are namespaced by embedding model.

    The vector dimension is part of the table schema, so a table written with a
    384-dimension MiniLM cannot be searched with a 768-dimension Google vector.
    Each (provider, model, dimension) therefore gets its own physical table —
    ``web_chunks__local__all-minilm-l6-v2__384`` — and the logical session name
    stays ``web_chunks`` for callers.
    """

    def __init__(
        self,
        db_path: str = "./srag_db",
        model_name: str = DEFAULT_LOCAL_MODEL,
        model=None,
        embedder: Optional[Embedder] = None,
        config=None,
    ):
        self.db_path = db_path
        self.db = lancedb.connect(db_path)

        if embedder is not None:
            self.embedder: Embedder = embedder
        elif config is not None:
            from srag.embeddings import build_embedder

            self.embedder = build_embedder(config, model=model)
        else:
            self.embedder = LocalEmbedder(model_name=model_name, model=model)

        # Kept for backwards compatibility: callers still reach for ``.model``.
        self.model = getattr(self.embedder, "_model", None)
        self.schema = self._schema_for(self.embedder.spec.dimension)

    # ── schema and naming ────────────────────────────────────────────────
    @staticmethod
    def _schema_for(dimension: int) -> pa.Schema:
        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("title", pa.string()),
                pa.field("content", pa.string()),
                pa.field("url", pa.string()),
                pa.field("timestamp", pa.string()),
                pa.field("author", pa.string()),
                pa.field("chunk_index", pa.int32()),
                pa.field("coherence_score", pa.float32()),
                pa.field("embedding", pa.list_(pa.float32(), dimension)),
            ]
        )

    @property
    def spec(self) -> EmbeddingSpec:
        return self.embedder.spec

    def physical_name(self, session: str) -> str:
        return f"{session}__{self.spec.slug}"

    # ── metadata sidecar ─────────────────────────────────────────────────
    def _meta_path(self) -> str:
        return os.path.join(self.db_path, META_FILENAME)

    def _read_meta(self) -> dict:
        try:
            with open(self._meta_path(), "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and isinstance(data.get("tables"), dict):
                return data
        except (OSError, ValueError):
            pass
        return {"version": 1, "tables": {}}

    def _write_meta(self, meta: dict) -> None:
        try:
            os.makedirs(self.db_path, exist_ok=True)
            tmp = f"{self._meta_path()}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2, sort_keys=True)
            os.replace(tmp, self._meta_path())
        except OSError as exc:  # pragma: no cover - disk full / read-only
            print(f"⚠️  Could not write {META_FILENAME}: {exc}")

    def _record_table(self, session: str, physical: str, rows: int) -> None:
        meta = self._read_meta()
        meta["tables"][physical] = {
            "session": session,
            "provider": self.spec.provider,
            "model": self.spec.model,
            "dimension": self.spec.dimension,
            "rows": rows,
            "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
        self._write_meta(meta)

    def delete_session(self, session: str) -> bool:
        """Drop a session's index and its metadata entry.

        Returns False when there is nothing to delete, so callers can answer
        404 instead of pretending the wipe succeeded.
        """
        physical = self.physical_name(session)
        tables = self._tables()
        if physical not in tables:
            # Indexes written before namespacing used the bare session name.
            if session in tables:
                physical = session
            else:
                return False

        self.db.drop_table(physical)
        meta = self._read_meta()
        if meta["tables"].pop(physical, None) is not None:
            self._write_meta(meta)
        return True

    def describe(self, session: str) -> Optional[dict]:
        """What model built a session's index — for CLI/MCP reporting."""
        physical = self.physical_name(session)
        entry = self._read_meta()["tables"].get(physical)
        if entry:
            return entry
        if session in self._tables():
            return {"session": session, "table": session, "legacy": True}
        return None

    # ── table helpers ────────────────────────────────────────────────────
    def _tables(self) -> List[str]:
        names = self.db.table_names()
        return list(names() if callable(names) else names)

    def _resolve_table_name(self, session: str) -> Optional[str]:
        """Prefer the namespaced table; fall back to a legacy unnamespaced one."""
        physical = self.physical_name(session)
        if physical in self._tables():
            return physical
        # Older indexes wrote straight to the session name with a 384-dim schema.
        if session in self._tables():
            table = self.db.open_table(session)
            dim = self._dimension_of(table)
            if dim is None or dim == self.spec.dimension:
                return session
            raise EmbeddingMismatchError(
                f"Index '{session}' was built with {dim}-dimension vectors but "
                f"'{self.spec.model}' produces {self.spec.dimension}. Re-ingest with the "
                f"original model, or index into a new session."
            )
        return None

    @staticmethod
    def _dimension_of(table) -> Optional[int]:
        try:
            field = table.schema.field("embedding")
            return int(field.type.list_size)
        except (KeyError, AttributeError, ValueError):
            return None

    def _get_or_create_table(self, session: str, force_new: bool = False):
        physical = self.physical_name(session)
        if force_new and physical in self._tables():
            self.db.drop_table(physical)
        if physical not in self._tables():
            return self.db.create_table(physical, schema=self.schema, mode="overwrite")
        return self.db.open_table(physical)

    def _embed(self, texts: List[str]):
        return self.embedder.embed(texts)

    # ── write ────────────────────────────────────────────────────────────
    def index_documents(
        self,
        docs: List[dict],
        table_name: str = "web_chunks",
        force_new: bool = False,
        min_coherence: float = 0.3,
    ):
        if not docs:
            return self._empty_stats(table_name)

        table = self._get_or_create_table(table_name, force_new=force_new)
        timestamp_prefix = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        rows = []

        for i, d in enumerate(docs):
            # Source validation — drop low coherence chunks
            coherence = d.get("coherence_score", 1.0)
            if coherence < min_coherence:
                print(f"⚠️  Dropping low coherence chunk ({coherence:.2f}): {d.get('title', '')}")
                continue

            source_url = d.get("url") or d.get("source") or "unknown_url"
            row_id = hashlib.md5(
                f"{source_url}_{timestamp_prefix}_{i}".encode()
            ).hexdigest()[:12]

            rows.append(
                {
                    "id": row_id,
                    "url": source_url,
                    "title": d.get("title") or "No Title",
                    "content": d.get("content") or "",
                    "timestamp": d.get("timestamp") or "",
                    "author": d.get("author") or "",
                    "chunk_index": d.get("chunk_index", 0),
                    "coherence_score": float(coherence),
                }
            )

        if not rows:
            print(f"⚠️  [{table_name}] All chunks dropped by coherence filter.")
            return self._empty_stats(table_name)

        embeddings = self._embed([r["content"] for r in rows])
        for r, emb in zip(rows, embeddings):
            r["embedding"] = list(emb)

        table.add(rows)
        physical = self.physical_name(table_name)
        self._record_table(table_name, physical, len(rows))
        print(
            f"📦 [{table_name}] Indexed {len(rows)} chunks "
            f"({self.spec.provider}/{self.spec.model}, {self.spec.dimension}d)."
        )
        return {
            "total_indexed": len(rows),
            "table_name": table_name,
            "physical_table": physical,
            "embedding": self.spec.to_dict(),
        }

    def _empty_stats(self, session: str) -> dict:
        """Returned instead of None, so callers can read counts without guarding."""
        return {
            "total_indexed": 0,
            "table_name": session,
            "physical_table": self.physical_name(session),
            "embedding": self.spec.to_dict(),
        }

    # ── read ─────────────────────────────────────────────────────────────
    def semantic_search(
        self,
        query: str,
        table_name: str = "web_chunks",
        k: int = 5,
    ) -> List[dict]:
        """Vector search with ID-based dedup — multiple chunks per page allowed."""
        resolved = self._resolve_table_name(table_name)
        if resolved is None:
            print(f"⚠️  Table '{table_name}' not found.")
            return []

        table = self.db.open_table(resolved)

        table_dim = self._dimension_of(table)
        if table_dim is not None and table_dim != self.spec.dimension:
            raise EmbeddingMismatchError(
                f"'{table_name}' holds {table_dim}-dimension vectors; the current embedder "
                f"({self.spec.provider}/{self.spec.model}) produces {self.spec.dimension}. "
                f"Switch the provider back, or re-ingest the session."
            )

        q_vec = self._embed([query])[0]

        results = (
            table.search(q_vec, vector_column_name="embedding")
            .metric("cosine")
            .limit(k * 2)
            .to_list()
        )

        seen_ids = set()
        deduped: List[dict] = []
        for r in results:
            rid = r.get("id", "")
            if rid not in seen_ids:
                seen_ids.add(rid)
                deduped.append(r)
            if len(deduped) >= k:
                break

        return deduped

    def query_session(self, query: str, session: str, k: int = 5) -> List[dict]:
        return self.semantic_search(query=query, table_name=session, k=k)

    # Bookkeeping tables SRag creates for itself; they are not user sessions.
    INTERNAL_TABLES = {"lexicon", "domain_reputation", "domain_filters", "sessions"}

    def list_sessions(self) -> List[str]:
        """Logical session names — namespaced physical tables are collapsed back."""
        sessions: List[str] = []
        meta = self._read_meta()["tables"]

        for physical in self._tables():
            entry = meta.get(physical)
            if entry and entry.get("session"):
                session = entry["session"]
            elif "__" in physical:
                # A table we did not write, but shaped like ours: drop the suffix.
                session = physical.rsplit("__", 3)[0]
            else:
                session = physical
            if session in self.INTERNAL_TABLES:
                continue
            if session not in sessions:
                sessions.append(session)

        return sessions
