"""Exercise the indexer foundation without torch or network access.

Converted from a print-and-exit script: it called `sys.exit()` at import time,
which crashed pytest collection with an INTERNALERROR instead of reporting a
test result. It also hardcoded the author's absolute path and registered a fake
`srag` module in sys.modules, which broke the other test modules.
"""

import os
import sys

import lancedb
import numpy as np
import pyarrow as pa
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srag.embeddings import EmbeddingMismatchError, EmbeddingSpec  # noqa: E402
from srag.indexer import SRagIndexer  # noqa: E402


class StubEmbedder:
    """Deterministic fake vectors, so the tests never touch a model."""

    def __init__(self, dimension, name):
        self._spec = EmbeddingSpec("local", name, dimension)

    @property
    def spec(self):
        return self._spec

    def embed(self, texts):
        out = []
        for text in texts:
            vector = np.zeros(self._spec.dimension, dtype="float32")
            for i, ch in enumerate(text[: self._spec.dimension]):
                vector[i] = (ord(ch) % 13) / 13.0
            vector[0] += 1.0
            out.append(vector.tolist())
        return out


DOCS = [
    {"url": "https://a.example", "title": "Alpha", "content": "alpha beta gamma", "coherence_score": 1.0},
    {"url": "https://b.example", "title": "Beta", "content": "delta epsilon zeta", "coherence_score": 0.9},
    {"url": "https://c.example", "title": "Weak", "content": "low coherence noise", "coherence_score": 0.1},
]


def test_index_search_and_namespacing(tmp_path):
    db = str(tmp_path / "foundation")

    # 1. index with a 384-dimension embedder
    small = SRagIndexer(db_path=db, embedder=StubEmbedder(384, "stub-384"))
    stats = small.index_documents(DOCS, table_name="web_chunks")
    assert stats["total_indexed"] == 2, "only coherent chunks kept"
    assert stats["physical_table"].endswith("__384"), "physical table is namespaced"
    assert stats["embedding"]["dimension"] == 384, "spec travels with the write"

    # 2. search and session listing
    hits = small.semantic_search("alpha beta", "web_chunks", k=2)
    assert len(hits) == 2, "search returns rows"
    assert all("content" in h for h in hits), "rows carry content"
    assert small.list_sessions() == ["web_chunks"], "logical session name only"
    assert small.describe("web_chunks")["dimension"] == 384, "describe reports the model"

    # 3. switch to a 768-dimension embedder, same session
    big = SRagIndexer(db_path=db, embedder=StubEmbedder(768, "stub-768"))
    stats2 = big.index_documents(DOCS, table_name="web_chunks")
    assert stats2["physical_table"].endswith("__768"), "new sibling table, not a rewrite"
    assert sorted(big.list_sessions()) == ["web_chunks"], "sessions still collapse to one name"
    assert small.describe("web_chunks")["dimension"] == 384, "old index untouched"
    assert len(big.semantic_search("alpha", "web_chunks", k=2)) == 2, "new embedder searches its own table"


def test_legacy_table_dimension_mismatch_is_refused(tmp_path):
    legacy_db = str(tmp_path / "legacy")
    db = lancedb.connect(legacy_db)
    legacy_schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("title", pa.string()),
            pa.field("content", pa.string()),
            pa.field("url", pa.string()),
            pa.field("timestamp", pa.string()),
            pa.field("author", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("coherence_score", pa.float32()),
            pa.field("embedding", pa.list_(pa.float32(), 384)),
        ]
    )
    db.create_table("old_session", schema=legacy_schema, mode="overwrite")

    # A 768-dimension embedder must refuse the 384-dimension legacy table.
    mismatched = SRagIndexer(db_path=legacy_db, embedder=StubEmbedder(768, "stub-768"))
    with pytest.raises(EmbeddingMismatchError):
        mismatched.semantic_search("anything", "old_session", k=1)

    # A matching one still works.
    matching = SRagIndexer(db_path=legacy_db, embedder=StubEmbedder(384, "stub-384"))
    assert matching.semantic_search("anything", "old_session", k=1) == []


def test_delete_session(tmp_path):
    """delete_session() is new in 1.3.9 — the HTTP endpoint used to fake success."""
    db = str(tmp_path / "delete")
    indexer = SRagIndexer(db_path=db, embedder=StubEmbedder(384, "stub-384"))
    indexer.index_documents(DOCS, table_name="web_chunks")
    assert indexer.list_sessions() == ["web_chunks"]

    assert indexer.delete_session("web_chunks") is True
    assert indexer.list_sessions() == [], "session is gone"
    assert indexer.describe("web_chunks") is None, "metadata entry is gone"

    assert indexer.delete_session("web_chunks") is False, "second delete reports nothing to do"
