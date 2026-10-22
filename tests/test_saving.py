import numpy as np
import pytest

from conftest import make_document
from chunking import build_config
from models import Chunk, Document
from saving import (
    bm25_index_path, chunking_config_path, chunks_path, embeddings_path, faiss_index_path, load_chunking_config,
    load_embeddings, parsed_path, read_jsonl, save_chunking_config, save_embeddings, write_jsonl,
)
from test_models import make_chunk


def test_paths_follow_the_naming_convention(tmp_path):
    names = [
        parsed_path(3, tmp_path), chunks_path(3, "fixed", tmp_path), chunking_config_path(3, "fixed", tmp_path),
        embeddings_path(3, "fixed", "sentence-transformers/all-MiniLM-L6-v2", tmp_path),
        faiss_index_path(3, "semantic", "all-mpnet-base-v2", tmp_path), bm25_index_path(3, "recursive", tmp_path),
    ]
    assert [p.name for p in names] == [
        "parsed_3.jsonl", "chunks_3_fixed.jsonl", "chunking_config_3_fixed.json",
        "embeddings_3_fixed_sentence-transformers-all-MiniLM-L6-v2.npz", "faiss_3_semantic_all-mpnet-base-v2.index",
        "bm25_3_recursive.pkl",
    ]


def test_jsonl_round_trip_validates_records(tmp_path):
    docs = [make_document(["One page."], source="a.pdf"), make_document(["Two.", "Pages."], source="b.pdf")]
    path = tmp_path / "docs.jsonl"
    assert write_jsonl(path, docs) == 2
    assert read_jsonl(path, Document) == docs
    path.write_text('{"id": "x"}\n')
    with pytest.raises(ValueError):
        read_jsonl(path, Chunk)


def test_chunking_config_round_trip(tmp_path):
    config = build_config("RECURSIVE", 512, 75, parser="pymupdf")
    save_chunking_config(tmp_path / "c.json", config)
    assert load_chunking_config(tmp_path / "c.json") == config


def test_embeddings_round_trip_and_detect_stale_caches(tmp_path):
    chunks = [make_chunk(), make_chunk()]
    vectors = np.arange(6, dtype=np.float32).reshape(2, 3)
    save_embeddings(tmp_path / "e.npz", chunks, vectors)
    assert np.array_equal(load_embeddings(tmp_path / "e.npz", chunks), vectors)
    with pytest.raises(ValueError, match="does not match"):
        load_embeddings(tmp_path / "e.npz", [chunks[1], chunks[0]])
    with pytest.raises(ValueError):
        save_embeddings(tmp_path / "e.npz", chunks, vectors[:1])
