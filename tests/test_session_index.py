import pytest

import embedding
from conftest import FakeEmbedder, write_pdf, paper_pages
from models import RetrievalConfig, RetrieverType
from query_phase import QuerySettings
from retrieval import MergedRetriever, RerankedRetriever, Retriever, _results
from session_index import Scope, UploadedFile, build_session_index, scoped_retriever
from test_models import make_chunk


@pytest.fixture(autouse=True)
def fake_models(monkeypatch):
    embedders = {}
    monkeypatch.setattr(embedding, "make_embedder",
                        lambda name, backend="sentence-transformers": embedders.setdefault(name, FakeEmbedder(name)))


def uploads(tmp_path) -> list[UploadedFile]:
    files = []
    for name, topic in [("grid.pdf", "impedance"), ("school.pdf", "enrollment")]:
        path = write_pdf(tmp_path / name, paper_pages(topic))
        files.append(UploadedFile(name, path.read_bytes()))
    return files


def config(method: str = "hybrid", reranker: str = "none") -> RetrievalConfig:
    return RetrievalConfig(chunk_method="recursive", embedding_model=None if method == "bm25" else "fake-a",
                           retrieval_method=method, top_k=3, reranker=reranker)


def test_session_index_chunks_uploads_in_memory_and_reports_bad_files(tmp_path):
    files = uploads(tmp_path) + [UploadedFile("broken.pdf", b"not a pdf")]
    index = build_session_index(files, "RECURSIVE", 40, 8, "fake-a")
    assert [d.metadata.source for d in index.documents] == ["grid.pdf", "school.pdf"]
    assert index.chunks and {c.metadata.source for c in index.chunks} == {"grid.pdf", "school.pdf"}
    assert set(index.errors) == {"broken.pdf"} and len(index.bm25) == len(index.chunks)
    assert index.vectors == {}                                      # vectors are built on first dense search
    assert not list(tmp_path.glob("**/chunks_*"))                   # nothing written to disk


@pytest.mark.parametrize("method", ["bm25", "dense", "hybrid"])
def test_uploads_scope_finds_the_matching_paper(tmp_path, method):
    index = build_session_index(uploads(tmp_path), "RECURSIVE", 40, 8, "fake-a")
    retriever = scoped_retriever(Scope.UPLOADS, config(method), QuerySettings(reranker="none"), index)
    results = retriever.retrieve("How is enrollment measured?", 3)
    assert results[0].chunk.metadata.source == "school.pdf"
    assert results[0].retriever_type is RetrieverType(method)
    assert ("fake-a" in index.vectors) == (method != "bm25")


def test_uploads_scope_needs_a_parsed_upload():
    with pytest.raises(ValueError, match="Upload at least one PDF"):
        scoped_retriever(Scope.UPLOADS, config(), QuerySettings(reranker="none"), None)


class Fixed(Retriever):
    retriever_type = RetrieverType.HYBRID

    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query, k):
        return _results(self.chunks, [(i, 10.0 - i) for i in range(min(k, len(self.chunks)))], self.retriever_type)


def test_both_scope_fuses_corpus_and_uploads_by_rank(tmp_path):
    index = build_session_index(uploads(tmp_path), "RECURSIVE", 40, 8, "fake-a")
    corpus = Fixed([make_chunk("corpus one", end_char=10), make_chunk("corpus two", end_char=10)])
    retriever = scoped_retriever(Scope.BOTH, config("bm25"), QuerySettings(reranker="none"), index, corpus)
    assert isinstance(retriever, MergedRetriever)
    results = retriever.retrieve("enrollment", 4)
    sources = [r.chunk.metadata.source for r in results]
    assert sources[0] in ("a.pdf", "school.pdf") and {"a.pdf", "school.pdf"} <= set(sources)   # rank 1 of each
    assert results[0].score == pytest.approx(1 / 61)


def test_both_scope_without_uploads_searches_the_corpus_only():
    corpus = Fixed([make_chunk("corpus one", end_char=10)])
    assert scoped_retriever(Scope.BOTH, config("bm25"), QuerySettings(reranker="none"), None, corpus) is corpus


def test_merged_retriever_sums_reciprocal_ranks_for_shared_chunks():
    shared, other = make_chunk("shared", end_char=6), make_chunk("other!", end_char=6)
    merged = MergedRetriever([Fixed([shared, other]), Fixed([shared])]).retrieve("q", 2)
    assert merged[0].chunk.id == shared.id and merged[0].score == pytest.approx(2 / 61)
    assert merged[1].score == pytest.approx(1 / 62)
    with pytest.raises(ValueError):
        MergedRetriever([])


def test_a_reranker_wraps_the_scoped_retriever(tmp_path, monkeypatch):
    import session_index

    class Length:
        name = "length"

        def rerank(self, query, results, k):
            return sorted(results, key=lambda r: -len(r.chunk.content))[:k]

    monkeypatch.setattr(session_index, "make_reranker", lambda name: Length())
    index = build_session_index(uploads(tmp_path), "RECURSIVE", 40, 8, "fake-a")
    retriever = scoped_retriever(Scope.UPLOADS, config(reranker="length"), QuerySettings(), index)
    assert isinstance(retriever, RerankedRetriever)
    results = retriever.retrieve("grid", 2)
    assert len(results[0].chunk.content) >= len(results[1].chunk.content)
