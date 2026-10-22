import numpy as np
import pytest

from conftest import FakeEmbedder
from indexing import BM25Index, FaissIndex, Index, build_index, make_vector_index, tokenize
from test_models import make_chunk

TEXTS = ["grid impedance estimation", "school enrollment rates", "active particle density", "kalman filter grid"]


def chunks():
    return [make_chunk(t, end_char=len(t)) for t in TEXTS]


def test_index_is_abstract():
    with pytest.raises(TypeError):
        Index()


def test_tokenize_keeps_numbers_whole():
    assert tokenize("RMSPE of 2.5% at 7,274 Hz") == ["rmspe", "of", "2.5", "at", "7,274", "hz"]


def test_faiss_index_finds_the_nearest_chunk_and_round_trips(tmp_path):
    embedder = FakeEmbedder()
    vectors = embedder.embed(TEXTS)
    index = build_index(FaissIndex(), chunks(), tmp_path / "f.index", vectors)
    query = embedder.embed(["enrollment rates in school"])[0]
    assert index.search(query, 1)[0][0] == 1
    loaded = FaissIndex.load(tmp_path / "f.index")
    assert len(loaded) == 4
    assert loaded.search(query, 4) == index.search(query, 4)
    assert len(index.search(query, 10)) == 4           # k larger than the index


def test_faiss_index_needs_one_vector_per_chunk_and_a_vector_query():
    with pytest.raises(ValueError):
        FaissIndex().build(chunks(), np.zeros((2, 4), dtype=np.float32))
    index = FaissIndex()
    index.build(chunks(), FakeEmbedder().embed(TEXTS))
    with pytest.raises(TypeError):
        index.search("text query", 1)


def test_bm25_index_ranks_keyword_matches_and_round_trips(tmp_path):
    index = build_index(BM25Index(), chunks(), tmp_path / "b.pkl")
    top = index.search("kalman grid", 2)
    assert [i for i, _ in top] == [3, 0]
    loaded = BM25Index.load(tmp_path / "b.pkl")
    assert len(loaded) == 4 and loaded.search("kalman grid", 2) == top
    with pytest.raises(TypeError):
        index.search(np.zeros(4), 1)


def test_make_vector_index():
    assert isinstance(make_vector_index("faiss"), FaissIndex)
    with pytest.raises(ValueError, match="LanceDB"):
        make_vector_index("LanceDB")
