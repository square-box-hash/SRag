import os
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pyarrow as pa
import lancedb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


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
    def __init__(
        self,
        db_path: str = "./srag_db",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model: Optional[SentenceTransformer] = None,
    ):
        self.db = lancedb.connect(db_path)
        self.model = SentenceTransformer(model_name)

        self.schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("title", pa.string()),
            pa.field("content", pa.string()),
            pa.field("url", pa.string()),
            pa.field("timestamp", pa.string()),
            pa.field("author", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("coherence_score", pa.float32()),
            pa.field("embedding", pa.list_(pa.float32(), 384)),
        ])

    def _get_or_create_table(self, table_name: str, force_new: bool = False):
        if force_new and table_name in self.db.table_names():
            self.db.drop_table(table_name)
        if table_name not in self.db.table_names():
            return self.db.create_table(table_name, schema=self.schema, mode="overwrite")
        return self.db.open_table(table_name)

    def _embed(self, texts: List[str]):
        return self.model.encode(texts, convert_to_numpy=True)

    def index_documents(
        self,
        docs: List[dict],
        table_name: str = "web_chunks",
        force_new: bool = False,
        min_coherence: float = 0.3,
    ):
        if not docs:
            return

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

            rows.append({
                "id": row_id,
                "url": source_url,
                "title": d.get("title") or "No Title",
                "content": d.get("content") or "",
                "timestamp": d.get("timestamp") or "",
                "author": d.get("author") or "",
                "chunk_index": d.get("chunk_index", 0),
                "coherence_score": float(coherence),
            })

        if not rows:
            print(f"⚠️  [{table_name}] All chunks dropped by coherence filter.")
            return

        embeddings = self._embed([r["content"] for r in rows])
        for r, emb in zip(rows, embeddings):
            r["embedding"] = emb.tolist()

        table.add(rows)
        print(f"📦 [{table_name}] Indexed {len(rows)} chunks.")
        return len(rows)

    def semantic_search(
        self,
        query: str,
        table_name: str = "web_chunks",
        k: int = 5,
    ) -> List[dict]:
        """Vector search with ID-based dedup — multiple chunks per page allowed."""
        if table_name not in self.db.table_names():
            print(f"⚠️  Table '{table_name}' not found.")
            return []

        table = self.db.open_table(table_name)
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

    def list_sessions(self) -> List[str]:
        return self.db.table_names()