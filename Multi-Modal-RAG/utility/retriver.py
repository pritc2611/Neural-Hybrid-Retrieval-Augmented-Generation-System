"""
retrieval.py — BM25 sparse index, HybridPineconeRetriever, RRF fusion.
"""
import re
import math
import asyncio
from typing import List, Dict, Tuple, Optional
import numpy as np
from langsmith import traceable


def _get_cfg():
    import utility.config as cfg
    return cfg


# =============================================================================
# BM25
# =============================================================================
class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.docs:     List[str]              = []
        self.metadata: List[Dict]             = []
        self._idf:     Dict[str, float]       = {}
        self._tf:      List[Dict[str, float]] = []
        self._avgdl:   float                  = 0.0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())
    
    def build(self, docs: List[str], metadata: List[Dict]):
        self.docs     = docs
        self.metadata = metadata
        tokenized     = [self.tokenize(d) for d in docs]
        self._avgdl   = float(np.mean([len(t) for t in tokenized])) if tokenized else 1.0
        self._tf = []
        df: Dict[str, int] = {}
        for tokens in tokenized:
            freq: Dict[str, float] = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            self._tf.append(freq)
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        N = len(docs)
        self._idf = {t: math.log((N - n + 0.5) / (n + 0.5) + 1) for t, n in df.items()}
    def query(self, q: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if not self.docs:
            return []
        tokens = self.tokenize(q)
        scores = np.zeros(len(self.docs))
        for t in tokens:
            if t not in self._idf:
                continue
            idf = self._idf[t]
            for i, tf in enumerate(self._tf):
                f     = tf.get(t, 0)
                dl    = sum(tf.values())
                denom = f + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return sorted(enumerate(scores.tolist()), key=lambda x: -x[1])[:top_k]


# Per-namespace registry
_bm25_indexes: Dict[str, BM25Index] = {}


def _get_bm25(namespace: str) -> BM25Index:
    if namespace not in _bm25_indexes:
        _bm25_indexes[namespace] = BM25Index()
    return _bm25_indexes[namespace]


def _update_bm25(namespace: str, docs: List[str], metadata: List[Dict]):
    idx      = _get_bm25(namespace)
    all_docs = idx.docs + docs
    all_meta = idx.metadata + metadata
    idx.build(all_docs, all_meta)
    print(f"✅ BM25 updated: ns='{namespace}', total={len(all_docs)}")


# =============================================================================
# RRF FUSION
# =============================================================================
def reciprocal_rank_fusion(
    dense_hits:  List[Tuple[str, float, Dict]],
    sparse_hits: List[Tuple[str, float, Dict]],
    k: int = 60,
) -> List[Tuple[str, float, Dict]]:
    cfg = _get_cfg()
    scores:   Dict[str, float] = {}
    meta_map: Dict[str, Dict]  = {}
    for rank, (doc_id, _, meta) in enumerate(dense_hits):
        scores[doc_id]   = scores.get(doc_id, 0.0) + cfg.DENSE_WEIGHT / (k + rank + 1)
        meta_map[doc_id] = meta
    for rank, (doc_id, _, meta) in enumerate(sparse_hits):
        scores[doc_id]   = scores.get(doc_id, 0.0) + cfg.SPARSE_WEIGHT / (k + rank + 1)
        meta_map[doc_id] = meta
    return [(did, sc, meta_map[did]) for did, sc in sorted(scores.items(), key=lambda x: -x[1])]


# =============================================================================
# HYBRID RETRIEVER
# =============================================================================
class HybridPineconeRetriever:
    def __init__(self, index, emb_client, top_k: int = 5, namespace: str = "__default__"):
        self.index             = index
        self.emb_client        = emb_client
        self.top_k             = top_k
        self.default_namespace = namespace

    async def __call__(self, query: str, namespace: Optional[str] = None):
        return await self.get_relevant_documents(query, namespace)
    @traceable(name="getting relevent documents")
    async def get_relevant_documents(self, query: str, namespace: Optional[str] = None):
        from langchain_core.documents import Document
        cfg = _get_cfg()
        if not query or not query.strip():
            return []

        ns    = namespace or cfg.current_namespace or self.default_namespace
        query = query.strip()

        dense_hits, sparse_hits = await asyncio.gather(
            self._dense_search(query, ns),
            self._sparse_search(query, ns),
        )

        fused = reciprocal_rank_fusion(dense_hits, sparse_hits)[: self.top_k]
        docs, seen = [], set()
        for doc_id, score, meta in fused:
            text = meta.get("text", "")
            if not text or text in seen:
                continue
            seen.add(text)
            docs.append(Document(
                page_content=text,
                metadata={"score": score, "source": meta.get("source", "unknown"), "id": doc_id},
            ))
        return docs

    async def _dense_search(self, query: str, namespace: str) -> List[Tuple[str, float, Dict]]:
        cfg = _get_cfg()
        try:
            vec    = await self.emb_client.embed(text=query)
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self.index.query,
                    vector=vec, top_k=self.top_k * 2,
                    include_metadata=True, namespace=namespace,
                ),
                timeout=8.0,
            )
            return [(m.id, m.score, m.metadata) for m in result.matches if m.metadata.get("text")]
        except Exception as e:
            print(f"⚠️  Dense search error: {e}")
            return []

    async def _sparse_search(self, query: str, namespace: str) -> List[Tuple[str, float, Dict]]:
        cfg = _get_cfg()
        try:
            bm25 = _get_bm25(namespace)
            if not bm25.docs:
                return []
            hits = await asyncio.get_event_loop().run_in_executor(
                cfg._thread_pool,
                lambda: bm25.query(query, top_k=self.top_k * 2),
            )
            return [
                (f"bm25-{idx}", score, {**bm25.metadata[idx], "text": bm25.docs[idx]})
                for idx, score in hits
            ]
        except Exception as e:
            print(f"⚠️  BM25 search error: {e}")
            return []