"""Hybrid retrieval: vector search + BM25 keyword search -> RRF merge -> rerank."""
import json
import re

import numpy as np
from qdrant_client.models import FieldCondition, Filter, MatchValue
from rank_bm25 import BM25Okapi

from . import config as C
from .models import client, embedder

_bm25_cache: dict[str, tuple] = {}
_reranker = None
_reranker_failed = False


def tokenize(s: str) -> list[str]:
    out = []
    for m in re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", s.lower()):
        out.append(m)  # keeps IDs like hr-45892-a whole
        if "-" in m or "_" in m:
            out.extend(re.split(r"[-_]", m))
    return out


def _bm25(doc_id: str):
    if doc_id not in _bm25_cache:
        chunks = json.loads((C.DOCS / f"{doc_id}.chunks.json").read_text())
        _bm25_cache[doc_id] = (BM25Okapi([tokenize(c["text"]) for c in chunks]), chunks)
    return _bm25_cache[doc_id]


def _rerank(query: str, cands: list[dict]) -> list[dict]:
    global _reranker, _reranker_failed
    if _reranker_failed or not cands:
        return cands
    try:
        if _reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            _reranker = TextCrossEncoder(model_name=C.RERANK_MODEL)
        scores = list(_reranker.rerank(query, [c["text"] for c in cands]))
        return [c for _, c in sorted(zip(scores, cands), key=lambda t: -t[0])]
    except Exception as e:
        print("Reranker unavailable, using fused order:", e)
        _reranker_failed = True
        return cands


def retrieve(doc_id: str, query: str, top_k: int = 5, pool: int = 20) -> list[dict]:
    # 1) semantic search
    qvec = next(iter(embedder().query_embed(query))).tolist()
    hits = client().query_points(
        C.COLLECTION, query=qvec, limit=pool, with_payload=True,
        query_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
    ).points
    vec_ranked = [h.payload for h in hits]

    # 2) keyword search
    bm, chunks = _bm25(doc_id)
    scores = bm.get_scores(tokenize(query))
    order = np.argsort(scores)[::-1][:pool]
    kw_ranked = [chunks[i] for i in order if scores[i] > 0]

    # 3) Reciprocal Rank Fusion
    fused: dict[str, dict] = {}
    for ranked in (vec_ranked, kw_ranked):
        for rank, c in enumerate(ranked):
            e = fused.setdefault(c["chunk_id"], {"chunk": c, "score": 0.0})
            e["score"] += 1.0 / (60 + rank + 1)
    cands = [e["chunk"] for e in sorted(fused.values(), key=lambda e: -e["score"])][:30]

    # 4) rerank, keep best few
    return _rerank(query, cands)[:top_k]
