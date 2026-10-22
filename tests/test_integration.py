"""
End-to-end ingestion on real arXiv papers from data/papers: parse → chunk (all
three methods) → embed (two models) → index, through the real files in a
temporary data directory. The default test uses deterministic fake embedders;
the slow test uses the real all-MiniLM-L6-v2 model.

Skipped when the papers or qrels.json aren't downloaded.
"""

import shutil

import numpy as np
import pytest

import embedding
from conftest import PAPERS_DIR, QRELS_FILE, FakeEmbedder
from embedding import make_embedder
from indexing import BM25Index, FaissIndex, tokenize
from models import Chunk, Document
from parser import qrels_doc_ids
from pipeline import IngestSettings, ingest
from saving import (
    bm25_index_path, chunks_path, embeddings_path, faiss_index_path, load_embeddings, parsed_path, read_jsonl,
)

pytestmark = pytest.mark.skipif(not (PAPERS_DIR.is_dir() and QRELS_FILE.exists()),
                                reason="data/papers or qrels.json not downloaded")
METHODS = {"FIXED": "fixed", "RECURSIVE": "recursive", "SEMANTIC": "semantic"}
CHUNK_SIZE = 512


@pytest.fixture(scope="module")
def small_papers(tmp_path_factory):
    """The two smallest papers that have qrels, copied to their own directory."""
    qrel_ids = qrels_doc_ids(QRELS_FILE)
    pdfs = sorted((p for p in PAPERS_DIR.glob("*.pdf") if p.stem in qrel_ids), key=lambda p: p.stat().st_size)[:2]
    d = tmp_path_factory.mktemp("papers")
    for p in pdfs:
        shutil.copy(p, d / p.name)
    return d


def settings_for(tmp_path, papers_dir, models, methods=tuple(METHODS)) -> IngestSettings:
    data = tmp_path / "data"
    data.mkdir()
    return IngestSettings(iteration=1, papers_dir=papers_dir, qrels_path=QRELS_FILE, max_papers=2, max_negatives=0,
                          methods=list(methods), models=list(models), chunk_size=CHUNK_SIZE, overlap=75,
                          semantic_model=models[0], data_dir=data, log_path=tmp_path / "log.jsonl")


def test_full_ingestion_grid_on_real_papers(tmp_path, small_papers, monkeypatch):
    embedders = {name: FakeEmbedder(name, dim=64, max_tokens=256) for name in ("fake-minilm", "fake-mpnet")}
    monkeypatch.setattr(embedding, "make_embedder", lambda name, backend="sentence-transformers": embedders[name])
    s = settings_for(tmp_path, small_papers, list(embedders))

    results = ingest(s)
    assert all(r.status == "ran" for r in results), [(r.label, r.error) for r in results if r.status != "ran"]

    documents = {d.id: d for d in read_jsonl(parsed_path(1, s.data_dir), Document)}
    assert len(documents) == 2
    for doc in documents.values():
        assert doc.metadata.has_qrels and doc.metadata.sections
        assert doc.metadata.sections[-1].end_char == len(doc.content)

    for key in METHODS.values():
        chunks = read_jsonl(chunks_path(1, key, s.data_dir), Chunk)
        assert {c.metadata.document_id for c in chunks} == set(documents)
        for c in chunks:
            doc, m = documents[c.metadata.document_id], c.metadata
            assert doc.content[m.start_char:m.end_char] == c.content
            assert 1 <= m.page_number <= m.end_page_number <= doc.metadata.page_count
            assert m.token_count <= CHUNK_SIZE and m.section_title is not None

        bm25 = BM25Index.load(bm25_index_path(1, key, s.data_dir))
        assert len(bm25) == len(chunks)
        probe = max(chunks, key=lambda c: len(set(tokenize(c.content))))
        assert chunks[bm25.search(" ".join(tokenize(probe.content)[:40]), 1)[0][0]].id == probe.id

        for model in embedders:
            vectors = load_embeddings(embeddings_path(1, key, model, s.data_dir), chunks)
            index = FaissIndex.load(faiss_index_path(1, key, model, s.data_dir))
            assert len(index) == len(chunks) == len(vectors)
            assert index.search(vectors[3], 1)[0][0] == 3
            # FAISS agrees with brute force (guards the KMP_DUPLICATE_LIB_OK workaround in config.py)
            for query in vectors[::7]:
                expected = np.argsort(-(vectors @ query), kind="stable")[:10]
                found = [i for i, _ in index.search(query, 10)]
                assert np.allclose(np.sort(vectors[found] @ query), np.sort(vectors[expected] @ query), atol=1e-5)


@pytest.mark.slow
def test_dense_search_with_real_minilm_finds_a_chunk_from_its_own_text(tmp_path, small_papers):
    model = "all-MiniLM-L6-v2"
    s = settings_for(tmp_path, small_papers, [model], methods=["FIXED"])
    assert all(r.status == "ran" for r in ingest(s))

    chunks = read_jsonl(chunks_path(1, "fixed", s.data_dir), Chunk)
    index = FaissIndex.load(faiss_index_path(1, "fixed", model, s.data_dir))
    target = chunks[len(chunks) // 2]
    query = make_embedder(model).embed([target.content[:300]])[0]
    assert chunks[index.search(query, 1)[0][0]].id == target.id
    assert np.allclose(np.linalg.norm(load_embeddings(embeddings_path(1, "fixed", model, s.data_dir)), axis=1), 1.0,
                       atol=1e-5)


def test_query_phase_on_real_papers_and_benchmark_queries(tmp_path, small_papers, monkeypatch):
    from conftest import FakeLLMClient, cite_first_source
    from generation import LLMGenerator
    from models import AnswerFile
    from query_phase import QuerySettings, run_query_phase
    from saving import load_model

    embedders = {name: FakeEmbedder(name, dim=64) for name in ("fake-minilm", "fake-mpnet")}
    monkeypatch.setattr(embedding, "make_embedder", lambda name, backend="sentence-transformers": embedders[name])
    ingest_settings = settings_for(tmp_path, small_papers, list(embedders))
    assert all(r.status == "ran" for r in ingest(ingest_settings))

    s = QuerySettings(iteration=1, models=list(embedders), reranker="none", top_k=5, query_count=2, query_abstractive=1,
                      query_source_mix="text:1.0", data_dir=ingest_settings.data_dir, log_path=tmp_path / "log.jsonl")
    paths = run_query_phase(s, LLMGenerator("vendor/fake", client=FakeLLMClient(cite_first_source)))
    assert len(paths) == 15
    for path in paths:
        answers = load_model(path, AnswerFile)
        for entry in answers.entries:
            assert entry.error is None and len(entry.retrieved) == 5
            citation = entry.response.citations[0]
            cited = next(c for c in entry.response.chunks_used if c.id == citation.chunk_id)
            assert citation.text_snippet in cited.content and citation.page_number == cited.metadata.page_number
