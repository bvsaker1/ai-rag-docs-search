"""
Query phase step 2: retrieve the top-K chunks for a query, optionally reranked.

    BM25Retriever    keyword search over a BM25Index
    DenseRetriever   embeds the query with the chunks' embedding model and searches a
                     FaissIndex (inner product of normalized vectors = cosine similarity)
    HybridRetriever  the top HYBRID_CANDIDATES from each, scores min-max normalized per
                     query and combined as alpha · dense + (1 - alpha) · bm25 (a chunk
                     missing from one list scores 0 there)

Rerankers (off unless RERANKER is set) reorder the first stage's top
RERANK_CANDIDATES and cut to TOP_K; each result's score becomes the reranker's:
    CrossEncoderReranker  a local sentence-transformers cross-encoder
    CohereReranker        Cohere rerank through OpenRouter's /rerank endpoint

Query vectors are cached per (model, query), so a query is embedded once per
embedding model and reused across chunking methods.
"""

import functools
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from api_retry import call_with_retries
from config import (
    COHERE_RERANK_MODEL, CROSS_ENCODER_MODEL, HYBRID_ALPHA, HYBRID_CANDIDATES, LLM_MAX_ATTEMPTS,
    LLM_REQUEST_TIMEOUT_SECONDS, OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
)
from embedding import Embedder
from indexing import BM25Index, FaissIndex, Index
from models import Chunk, RetrievalResult, RetrieverType


class Retriever(ABC):
    retriever_type: RetrieverType

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        """The top `k` chunks for `query`, best first."""


def _results(chunks: list[Chunk], hits: list[tuple[int, float]], kind: RetrieverType) -> list[RetrievalResult]:
    return [RetrievalResult(chunk=chunks[i], score=score, retriever_type=kind) for i, score in hits]


class BM25Retriever(Retriever):
    retriever_type = RetrieverType.BM25

    def __init__(self, chunks: list[Chunk], index: Index) -> None:
        if len(index) != len(chunks):
            raise ValueError(f"BM25 index has {len(index)} entries for {len(chunks)} chunks")
        self.chunks, self.index = chunks, index

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        return _results(self.chunks, self.index.search(query, k), self.retriever_type)


_query_vectors: dict[tuple[str, str], np.ndarray] = {}


def clear_query_vectors() -> None:
    _query_vectors.clear()


def query_vector(embedder: Embedder, query: str) -> np.ndarray:
    """The query's embedding, computed once per (model, query)."""
    key = (embedder.name, query)
    if key not in _query_vectors:
        _query_vectors[key] = embedder.embed([query])[0]
    return _query_vectors[key]


class DenseRetriever(Retriever):
    retriever_type = RetrieverType.DENSE

    def __init__(self, chunks: list[Chunk], index: Index, embedder: Embedder) -> None:
        if len(index) != len(chunks):
            raise ValueError(f"Vector index has {len(index)} entries for {len(chunks)} chunks")
        self.chunks, self.index, self.embedder = chunks, index, embedder

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        return _results(self.chunks, self.index.search(query_vector(self.embedder, query), k), self.retriever_type)


def min_max(scores: dict) -> dict:
    """Scores rescaled to [0, 1]; all equal → all 1."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    return {k: 1.0 if hi == lo else (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRetriever(Retriever):
    retriever_type = RetrieverType.HYBRID

    def __init__(self, lexical: Retriever, dense: Retriever, alpha: float = HYBRID_ALPHA,
                 candidates: int = HYBRID_CANDIDATES) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"HYBRID_ALPHA must be between 0 and 1, got {alpha}")
        self.lexical, self.dense, self.alpha, self.candidates = lexical, dense, alpha, candidates

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        depth = max(k, self.candidates)
        lexical = {r.chunk.id: r for r in self.lexical.retrieve(query, depth)}
        dense = {r.chunk.id: r for r in self.dense.retrieve(query, depth)}
        lex_norm = min_max({cid: r.score for cid, r in lexical.items()})
        dense_norm = min_max({cid: r.score for cid, r in dense.items()})
        fused = {cid: self.alpha * dense_norm.get(cid, 0.0) + (1 - self.alpha) * lex_norm.get(cid, 0.0)
                 for cid in lexical.keys() | dense.keys()}
        # Ties break by dense rank, then BM25 rank, so results are deterministic.
        order = {cid: i for i, cid in enumerate([*dense, *(c for c in lexical if c not in dense)])}
        top = sorted(fused, key=lambda cid: (-fused[cid], order[cid]))[:k]
        return [RetrievalResult(chunk=(dense.get(cid) or lexical[cid]).chunk, score=fused[cid],
                                retriever_type=self.retriever_type) for cid in top]


# ── Reranking ─────────────────────────────────────────────────────────────────

class Reranker(ABC):
    name: str

    @abstractmethod
    def score(self, query: str, documents: list[str]) -> list[float]:
        """A relevance score per document, in document order."""

    def rerank(self, query: str, results: list[RetrievalResult], k: int) -> list[RetrievalResult]:
        """`results` reordered by relevance to `query` and cut to `k`; scores become the reranker's."""
        if not results:
            return []
        scores = self.score(query, [r.chunk.content for r in results])
        order = sorted(range(len(results)), key=lambda i: -scores[i])[:k]
        return [results[i].model_copy(update={"score": float(scores[i])}) for i in order]


class CrossEncoderReranker(Reranker):
    name = "cross-encoder"

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [float(s) for s in self.model.predict([(query, d) for d in documents], show_progress_bar=False)]


# post(url, headers, payload) -> parsed JSON response; raises on HTTP errors.
Poster = Callable[[str, dict, dict], dict]


def _httpx_post(url: str, headers: dict, payload: dict) -> dict:
    import httpx

    response = httpx.post(url, headers=headers, json=payload, timeout=LLM_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


class CohereReranker(Reranker):
    """Cohere rerank via OpenRouter's /rerank endpoint, retried as in api_retry."""

    name = "cohere"

    def __init__(self, model: str = COHERE_RERANK_MODEL, post: Optional[Poster] = None,
                 max_attempts: int = LLM_MAX_ATTEMPTS, sleep: Callable[[float], None] = time.sleep) -> None:
        if post is None and not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to your environment or .env file.")
        self.model, self._post, self._max_attempts, self._sleep = model, post or _httpx_post, max_attempts, sleep

    def score(self, query: str, documents: list[str]) -> list[float]:
        payload = {"model": self.model, "query": query, "documents": documents}
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}

        def call() -> list[float]:
            response = self._post(f"{OPENROUTER_BASE_URL}/rerank", headers, payload)
            scores = {int(r["index"]): float(r["relevance_score"]) for r in response["results"]}
            if sorted(scores) != list(range(len(documents))):
                raise ValueError(f"rerank returned {len(scores)} results for {len(documents)} documents")
            return [scores[i] for i in range(len(documents))]

        return call_with_retries(call, self._max_attempts, sleep=self._sleep)


class RerankedRetriever(Retriever):
    """A first-stage retriever whose top `candidates` are reranked and cut to k."""

    def __init__(self, first_stage: Retriever, reranker: Reranker, candidates: int) -> None:
        self.first_stage, self.reranker, self.candidates = first_stage, reranker, candidates
        self.retriever_type = first_stage.retriever_type

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        return self.reranker.rerank(query, self.first_stage.retrieve(query, max(k, self.candidates)), k)


RERANKERS: dict[str, type[Reranker]] = {"cross-encoder": CrossEncoderReranker, "cohere": CohereReranker}


@functools.cache
def make_reranker(name: str) -> Optional[Reranker]:
    """None for "none"; one shared instance per reranker otherwise."""
    if name.lower() == "none":
        return None
    if name.lower() not in RERANKERS:
        raise ValueError(f"Unknown reranker {name!r}; expected none or one of {', '.join(RERANKERS)}")
    return RERANKERS[name.lower()]()


# ── Merging ───────────────────────────────────────────────────────────────────

class MergedRetriever(Retriever):
    """Several retrievers over different chunk collections (e.g. the corpus and a session's
    uploaded PDFs), fused by reciprocal rank: score = Σ 1 / (rrf_k + rank). Their own scores
    aren't comparable — BM25 depends on each collection's statistics — but ranks are."""

    def __init__(self, retrievers: list[Retriever], rrf_k: int = 60) -> None:
        if not retrievers:
            raise ValueError("MergedRetriever needs at least one retriever")
        self.retrievers, self.rrf_k = retrievers, rrf_k
        self.retriever_type = retrievers[0].retriever_type

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        fused: dict = {}
        for retriever in self.retrievers:
            for rank, r in enumerate(retriever.retrieve(query, k), 1):
                score, _ = fused.get(r.chunk.id, (0.0, r))
                fused[r.chunk.id] = (score + 1.0 / (self.rrf_k + rank), r)
        ranked = sorted(fused.values(), key=lambda item: -item[0])[:k]
        return [r.model_copy(update={"score": score}) for score, r in ranked]


# ── Factory ───────────────────────────────────────────────────────────────────

def build_retriever(method: str | RetrieverType, chunks: list[Chunk], bm25_index: Optional[Index] = None,
                    faiss_index: Optional[Index] = None, embedder: Optional[Embedder] = None,
                    alpha: float = HYBRID_ALPHA, candidates: int = HYBRID_CANDIDATES) -> Retriever:
    """The first-stage retriever for a method name (bm25 | dense | hybrid) over loaded indexes."""
    kind = RetrieverType(method.lower() if isinstance(method, str) else method)

    def bm25() -> Retriever:
        if bm25_index is None:
            raise ValueError("BM25 retrieval needs a BM25 index")
        return BM25Retriever(chunks, bm25_index)

    def dense() -> Retriever:
        if embedder is None or faiss_index is None:
            raise ValueError("Dense retrieval needs the chunks' embedding model and vector index")
        return DenseRetriever(chunks, faiss_index, embedder)

    return {RetrieverType.BM25: bm25, RetrieverType.DENSE: dense,
            RetrieverType.HYBRID: lambda: HybridRetriever(bm25(), dense(), alpha, candidates)}[kind]()


def make_retriever(method: str | RetrieverType, chunks: list[Chunk], bm25_path: Optional[Path] = None,
                   faiss_path: Optional[Path] = None, embedder: Optional[Embedder] = None,
                   alpha: float = HYBRID_ALPHA, candidates: int = HYBRID_CANDIDATES,
                   reranker: Optional[Reranker] = None, rerank_candidates: int = 20) -> Retriever:
    """The retriever for a method name over saved indexes, wrapped in `reranker` when given.
    Only the indexes the method needs are loaded."""
    kind = RetrieverType(method.lower() if isinstance(method, str) else method)
    if kind is not RetrieverType.BM25 and embedder is None:
        raise ValueError("Dense retrieval needs the chunks' embedding model")
    bm25_index = BM25Index.load(bm25_path) if kind is not RetrieverType.DENSE else None
    faiss_index = FaissIndex.load(faiss_path) if kind is not RetrieverType.BM25 else None
    retriever = build_retriever(kind, chunks, bm25_index, faiss_index, embedder, alpha, candidates)
    return RerankedRetriever(retriever, reranker, rerank_candidates) if reranker else retriever
