"""Embedding providers.

The vector dimension is a property of the *model*, not of the database, and a
LanceDB table bakes that dimension into its schema. So the provider, the model id
and the dimension all have to travel together: every embedder reports them as an
``EmbeddingSpec``, and the indexer namespaces its tables by that spec. Switching
from a local MiniLM (384) to Google's text-embedding-004 (768) then creates a
sibling table instead of silently corrupting the existing one.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Protocol

DEFAULT_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GOOGLE_MODEL = "text-embedding-004"

# Known output dimensions, so a caller can name a model without also naming a dim.
GOOGLE_DIMENSIONS = {
    "text-embedding-004": 768,
    "text-embedding-005": 768,
    "gemini-embedding-001": 3072,
}


@dataclass(frozen=True)
class EmbeddingSpec:
    """Everything that has to match for a query to be comparable to an index."""

    provider: str  # "local" | "google"
    model: str
    dimension: int

    @property
    def slug(self) -> str:
        """Filesystem- and table-safe identifier, e.g. ``local__minilm-l6-v2__384``."""
        short = self.model.split("/")[-1].lower()
        short = re.sub(r"[^a-z0-9]+", "-", short).strip("-")
        return f"{self.provider}__{short}__{self.dimension}"

    def to_dict(self) -> dict:
        return {"provider": self.provider, "model": self.model, "dimension": self.dimension}


class EmbeddingMismatchError(ValueError):
    """Raised when a query is embedded with a different model than the index used."""


class Embedder(Protocol):
    @property
    def spec(self) -> EmbeddingSpec: ...

    def embed(self, texts: List[str]) -> List[List[float]]: ...


class LocalEmbedder:
    """sentence-transformers, running on the machine. No key, no network."""

    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL, model=None):
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
        self._model = model

        dimension = getattr(model, "get_sentence_embedding_dimension", lambda: None)()
        if not dimension:
            raise ValueError(
                f"Could not determine the embedding dimension of '{model_name}'. "
                "Pass an explicit dimension if this model does not report one."
            )
        self._spec = EmbeddingSpec("local", model_name, int(dimension))

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return [vector.tolist() for vector in vectors]


class GoogleEmbedder:
    """Google Generative Language embeddings over REST.

    Deliberately dependency-free: httpx is already required by SRag, so adding
    Google support costs no new package. The text does leave the machine, which
    is the trade you are making — keep ``LocalEmbedder`` the default.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        model: str = DEFAULT_GOOGLE_MODEL,
        api_key: Optional[str] = None,
        dimension: Optional[int] = None,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ):
        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Google embeddings need an API key: set GOOGLE_API_KEY or GEMINI_API_KEY, "
                "or pass api_key=."
            )
        self._key = key
        self._model = model
        self._task_type = task_type
        self._requested_dimension = dimension

        resolved = dimension or GOOGLE_DIMENSIONS.get(model)
        if not resolved:
            resolved = self._probe_dimension()
        self._spec = EmbeddingSpec("google", model, int(resolved))

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def _probe_dimension(self) -> int:
        """One tiny request, so an unknown model id still gets a correct schema."""
        vectors = self._embed_batch(["dimension probe"])
        return len(vectors[0])

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        import httpx

        payload_requests = []
        for text in texts:
            entry = {
                "model": f"models/{self._model}",
                "content": {"parts": [{"text": text}]},
                "taskType": self._task_type,
            }
            if self._requested_dimension:
                entry["outputDimensionality"] = self._requested_dimension
            payload_requests.append(entry)

        response = httpx.post(
            f"{self.BASE_URL}/models/{self._model}:batchEmbedContents",
            params={"key": self._key},
            json={"requests": payload_requests},
            timeout=60.0,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Google embeddings failed ({response.status_code}): {response.text[:300]}"
            )
        return [item["values"] for item in response.json()["embeddings"]]

    def embed(self, texts: List[str]) -> List[List[float]]:
        # The endpoint caps a batch at 100 inputs.
        out: List[List[float]] = []
        for start in range(0, len(texts), 100):
            out.extend(self._embed_batch(texts[start : start + 100]))
        return out


def build_embedder(config=None, model=None) -> Embedder:
    """Turn an ``SRagConfig`` into a concrete embedder.

    ``model`` lets the orchestrator share one already-loaded SentenceTransformer
    across the chunker and the indexer instead of loading torch twice.
    """
    provider = (getattr(config, "embedding_provider", "") or "local").lower()
    model_name = getattr(config, "embedding_model", "") or ""
    dimension = getattr(config, "embedding_dimension", 0) or None
    api_key = getattr(config, "google_api_key", "") or None

    if provider == "google":
        return GoogleEmbedder(
            model=model_name or DEFAULT_GOOGLE_MODEL,
            api_key=api_key,
            dimension=dimension,
        )

    if provider in ("local", "", "sentence-transformers"):
        return LocalEmbedder(model_name=model_name or DEFAULT_LOCAL_MODEL, model=model)

    raise ValueError(
        f"Unknown embedding provider '{provider}'. Use 'local' or 'google'."
    )
