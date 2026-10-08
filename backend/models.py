"""Lazily loaded shared objects: Qdrant client and embedding model."""
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from . import config as C

_client = None
_embedder = None


def client() -> QdrantClient:
    global _client
    if _client is None:
        # Local on-disk mode: no Docker needed. For production, point this at a Qdrant server.
        _client = QdrantClient(path=str(C.QDRANT_PATH))
        if not _client.collection_exists(C.COLLECTION):
            _client.create_collection(
                C.COLLECTION,
                vectors_config=VectorParams(size=C.EMBED_DIM, distance=Distance.COSINE),
            )
    return _client


def embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=C.EMBED_MODEL)
    return _embedder
