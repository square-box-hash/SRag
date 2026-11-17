"""Exercise the indexer foundation without torch or network access."""

import shutil
import sys

import numpy as np
import pyarrow as pa
import lancedb

sys.path.insert(0, "/scratch/work/srag")

# Register a package stub so importing submodules does not pull in torch via __init__.
import types
_pkg = types.ModuleType("srag")
_pkg.__path__ = ["/scratch/work/srag/srag"]
sys.modules["srag"] = _pkg

from srag.embeddings import EmbeddingMismatchError, EmbeddingSpec
from srag.indexer import SRagIndexer


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

DB = "/tmp/srag_foundation_test"
shutil.rmtree(DB, ignore_errors=True)

failures = []


def check(label, condition, detail=""):
    print(("  ok   " if condition else "  FAIL ") + label + (f"  [{detail}]" if detail else ""))
    if not condition:
        failures.append(label)


print("1. index with a 384-dimension embedder")
small = SRagIndexer(db_path=DB, embedder=StubEmbedder(384, "stub-384"))
stats = small.index_documents(DOCS, table_name="web_chunks")
check("only coherent chunks kept", stats["total_indexed"] == 2, str(stats["total_indexed"]))
check("physical table is namespaced", stats["physical_table"].endswith("__384"), stats["physical_table"])
check("spec travels with the write", stats["embedding"]["dimension"] == 384)

print("2. search and session listing")
hits = small.semantic_search("alpha beta", "web_chunks", k=2)
check("search returns rows", len(hits) == 2, str(len(hits)))
check("rows carry content", all("content" in h for h in hits))
check("logical session name only", small.list_sessions() == ["web_chunks"], str(small.list_sessions()))
check("describe reports the model", small.describe("web_chunks")["dimension"] == 384)

print("3. switch to a 768-dimension embedder, same session")
big = SRagIndexer(db_path=DB, embedder=StubEmbedder(768, "stub-768"))
stats2 = big.index_documents(DOCS, table_name="web_chunks")
check("new sibling table, not a rewrite", stats2["physical_table"].endswith("__768"), stats2["physical_table"])
check("sessions still collapse to one name", sorted(big.list_sessions()) == ["web_chunks"], str(big.list_sessions()))
check("old index untouched", small.describe("web_chunks")["dimension"] == 384)
check("new embedder searches its own table", len(big.semantic_search("alpha", "web_chunks", k=2)) == 2)

print("4. legacy table with a mismatched dimension is refused, not corrupted")
legacy_db = "/tmp/srag_legacy_test"
shutil.rmtree(legacy_db, ignore_errors=True)
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

mismatched = SRagIndexer(db_path=legacy_db, embedder=StubEmbedder(768, "stub-768"))
try:
    mismatched.semantic_search("anything", "old_session", k=1)
    check("legacy mismatch raises", False, "no error raised")
except EmbeddingMismatchError as exc:
    check("legacy mismatch raises", True, str(exc)[:60] + "…")

print("5. legacy table with a matching dimension still works")
matching = SRagIndexer(db_path=legacy_db, embedder=StubEmbedder(384, "stub-384"))
check("matching legacy table resolves", matching.semantic_search("anything", "old_session", k=1) == [])

print()
print("FAILURES:", failures if failures else "none")
sys.exit(1 if failures else 0)
