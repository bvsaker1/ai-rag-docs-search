import numpy as np
import pytest

import retrieval
from conftest import FakeEmbedder
from indexing import BM25Index, FaissIndex, build_index
from models import RetrieverType
from retrieval import (
    BM25Retriever, CohereReranker, DenseRetriever, HybridRetriever, Reranker, RerankedRetriever, Retriever,
    make_reranker, make_retriever, min_max, query_vector,
)
from test_models import make_chunk

TEXTS = ["grid impedance estimation with recursive least squares", "school enrollment rates in surveys",
         "active particle density fluctuations", "kalman filter for grid voltage", "imputation of missing data"]


@pytest.fixture
def chunks():
    return [make_chunk(t, end_char=len(t)) for t in TEXTS]


@pytest.fixture
def indexes(tmp_path, chunks):
    embedder = FakeEmbedder("fake-retrieval")
    build_index(BM25Index(), chunks, tmp_path / "b.pkl")
    build_index(FaissIndex(), chunks, tmp_path / "f.index", embedder.embed(TEXTS))
    return tmp_path / "b.pkl", tmp_path / "f.index", embedder


def test_retriever_and_reranker_are_abstract():
    for cls in (Retriever, Reranker):
        with pytest.raises(TypeError):
            cls()


def test_bm25_and_dense_return_typed_ranked_results(chunks, indexes):
    bm25_path, faiss_path, embedder = indexes
    bm25 = BM25Retriever(chunks, BM25Index.load(bm25_path)).retrieve("kalman grid", 2)
    assert [r.chunk.content for r in bm25] == [TEXTS[3], TEXTS[0]]
    assert all(r.retriever_type is RetrieverType.BM25 for r in bm25)
    dense = DenseRetriever(chunks, FaissIndex.load(faiss_path), embedder).retrieve("school enrollment", 1)
    assert dense[0].chunk.content == TEXTS[1] and dense[0].retriever_type is RetrieverType.DENSE


def test_retrievers_reject_indexes_built_for_other_chunks(chunks, indexes):
    with pytest.raises(ValueError, match="entries"):
        BM25Retriever(chunks[:2], BM25Index.load(indexes[0]))


def test_query_vectors_are_cached_per_model_and_query():
    embedder = FakeEmbedder("fake-cache")
    first = query_vector(embedder, "grid")
    assert query_vector(embedder, "grid") is first and embedder.calls == 1
    query_vector(embedder, "other")
    assert embedder.calls == 2


def test_min_max():
    assert min_max({"a": 2.0, "b": 4.0, "c": 3.0}) == {"a": 0.0, "b": 1.0, "c": 0.5}
    assert min_max({"a": 7.0}) == {"a": 1.0} and min_max({}) == {}


class Fixed(Retriever):
    """Returns fixed (chunk index, score) pairs."""

    def __init__(self, chunks, hits, kind=RetrieverType.BM25):
        self.chunks, self.hits, self.retriever_type = chunks, hits, kind

    def retrieve(self, query, k):
        return retrieval._results(self.chunks, self.hits[:k], self.retriever_type)


def test_hybrid_combines_normalized_scores_with_alpha(chunks):
    lexical = Fixed(chunks, [(0, 10.0), (1, 6.0), (2, 2.0)])
    dense = Fixed(chunks, [(2, 0.9), (3, 0.5), (0, 0.1)], RetrieverType.DENSE)
    results = HybridRetriever(lexical, dense, alpha=0.5, candidates=3).retrieve("q", 4)
    # bm25 norm: c0 1, c1 .5, c2 0 · dense norm: c2 1, c3 .5, c0 0 · ties break by dense rank, then BM25 rank
    assert [(r.chunk.content, round(r.score, 3)) for r in results] == [
        (TEXTS[2], 0.5), (TEXTS[0], 0.5), (TEXTS[3], 0.25), (TEXTS[1], 0.25)]
    assert all(r.retriever_type is RetrieverType.HYBRID for r in results)
    dense_only = HybridRetriever(lexical, dense, alpha=1.0, candidates=3).retrieve("q", 1)
    assert dense_only[0].chunk.content == TEXTS[2]
    with pytest.raises(ValueError, match="HYBRID_ALPHA"):
        HybridRetriever(lexical, dense, alpha=1.5)


class LengthReranker(Reranker):
    name = "length"

    def score(self, query, documents):
        return [float(len(d)) for d in documents]


def test_reranked_retriever_reorders_first_stage_candidates_and_cuts_to_k(chunks):
    first = Fixed(chunks, [(i, 1.0) for i in range(5)])
    results = RerankedRetriever(first, LengthReranker(), candidates=5).retrieve("q", 2)
    assert [r.chunk.content for r in results] == sorted(TEXTS, key=len, reverse=True)[:2]
    assert results[0].score == float(len(results[0].chunk.content))
    assert results[0].retriever_type is RetrieverType.BM25


def test_cohere_reranker_posts_to_openrouter_and_maps_scores_back():
    calls = []

    def post(url, headers, payload):
        calls.append((url, payload))
        return {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]}

    reranker = CohereReranker(model="cohere/rerank-v3.5", post=post)
    assert reranker.score("q", ["a", "b"]) == [0.2, 0.9]
    assert calls[0][0].endswith("/rerank") and calls[0][1]["documents"] == ["a", "b"]

    bad = CohereReranker(post=lambda *a: {"results": [{"index": 0, "relevance_score": 1.0}]}, max_attempts=1)
    with pytest.raises(ValueError, match="1 results for 2"):
        bad.score("q", ["a", "b"])


def test_make_reranker_is_off_by_default():
    assert make_reranker("none") is None and make_reranker("NONE") is None
    assert make_reranker("cross-encoder").name == "cross-encoder"
    with pytest.raises(ValueError, match="Unknown reranker"):
        make_reranker("colbert")


def test_make_retriever_builds_each_method_from_saved_indexes(chunks, indexes):
    bm25_path, faiss_path, embedder = indexes
    kinds = {m: make_retriever(m, chunks, bm25_path, faiss_path, embedder) for m in ("bm25", "dense", "hybrid")}
    assert {m: r.retriever_type.value for m, r in kinds.items()} == {"bm25": "bm25", "dense": "dense", "hybrid": "hybrid"}
    reranked = make_retriever("hybrid", chunks, bm25_path, faiss_path, embedder, reranker=LengthReranker())
    assert isinstance(reranked, RerankedRetriever) and reranked.retriever_type is RetrieverType.HYBRID
    with pytest.raises(ValueError, match="embedding model"):
        make_retriever("dense", chunks, bm25_path, faiss_path)
    with pytest.raises(ValueError):
        make_retriever("tfidf", chunks, bm25_path)


@pytest.mark.slow
def test_cross_encoder_ranks_the_relevant_passage_first():
    reranker = make_reranker("cross-encoder")
    scores = reranker.score("How is grid impedance estimated?", [TEXTS[1], TEXTS[0]])
    assert scores[1] > scores[0]
