from dataclasses import dataclass
from typing import List

import lancedb
from sentence_transformers import SentenceTransformer


@dataclass
class IndexedChunk:
    id: str
    url: str
    title: str
    content: str
    timestamp: str
    author: str | None


class SRagIndexer:
    def __init__(
        self,
        db_path: str = "./srag_db",
        table_name: str = "web_chunks",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.db = lancedb.connect(db_path)
        self.table_name = table_name
        self.model = SentenceTransformer(model_name)

        if table_name not in self.db.table_names():
            self.table = self.db.create_table(
                table_name,
                data=[],
                mode="overwrite",
            )
        else:
            self.table = self.db.open_table(table_name)

    def _embed(self, texts: List[str]):
        return self.model.encode(texts, convert_to_numpy=True)

    def index_documents(self, docs: List[dict]):
        rows = []
        for i, d in enumerate(docs):
            row_id = f"{d['source']}#{i}"
            rows.append(
                {
                    "id": row_id,
                    "url": d["source"],
                    "title": d["title"],
                    "content": d["content"],
                    "timestamp": d.get("timestamp", ""),
                    "author": d.get("author"),
                }
            )

        embeddings = self._embed([r["content"] for r in rows])
        for r, emb in zip(rows, embeddings):
            r["vector"] = emb

        self.table.add(rows)

    def semantic_search(self, query: str, k: int = 5) -> List[dict]:
        q_vec = self._embed([query])[0]
        results = (
            self.table.search(q_vec)
            .metric("cosine")
            .limit(k)
            .to_list()
        )
        return results
