import numpy as np
import pytest

from config import Logger
from conftest import FakeEmbedder
from embedding import Embedder, SentenceTransformerEmbedder, embed_chunks, make_embedder, truncated_count
from test_models import make_chunk


def test_embedder_is_abstract():
    with pytest.raises(TypeError):
        Embedder()


def test_truncated_count_compares_token_counts_to_the_model_limit():
    assert truncated_count(FakeEmbedder(max_tokens=3), ["a b", "a b c d", "a b c"]) == 1
    assert truncated_count(FakeEmbedder(), ["a b c d"]) is None


def test_embed_chunks_returns_one_row_per_chunk_and_logs_truncation(tmp_path):
    chunks = [make_chunk("ab cd"), make_chunk("ef gh")]
    log_file = tmp_path / "log.jsonl"
    log = Logger(1, "embedding", path=log_file)
    vectors = embed_chunks(chunks, FakeEmbedder(max_tokens=1), log)
    log.close()
    assert vectors.shape == (2, 32)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert '"chunks_truncated"' in log_file.read_text() and '"truncated": 2' in log_file.read_text()


class StubModel:
    max_seq_length = 4

    def __init__(self) -> None:
        self.kwargs = None

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts, **kwargs):
        self.kwargs = kwargs
        return np.ones((len(texts), 3), dtype=np.float64)

    def tokenizer(self, texts, **kwargs):
        return {"input_ids": [t.split() + ["[CLS]", "[SEP]"] for t in texts]}


def test_sentence_transformer_embedder_normalizes_and_counts_model_tokens():
    embedder = SentenceTransformerEmbedder("stub")
    embedder._model = StubModel()
    vectors = embedder.embed(["a", "b"])
    assert vectors.dtype == np.float32 and vectors.shape == (2, 3)
    assert embedder.model.kwargs["normalize_embeddings"] is True
    assert embedder.embed([]).shape == (0, 3)
    assert embedder.count_tokens(["a b c"]) == [5]
    assert truncated_count(embedder, ["a b", "a b c"]) == 1


def test_make_embedder_shares_instances_and_rejects_unknown_backends():
    assert make_embedder("all-MiniLM-L6-v2") is make_embedder("all-MiniLM-L6-v2")
    with pytest.raises(ValueError, match="backend"):
        make_embedder("x", backend="openai")


@pytest.mark.slow
def test_minilm_embeds_normalized_384_dim_vectors_with_256_token_limit():
    embedder = make_embedder("all-MiniLM-L6-v2")
    vectors = embedder.embed(["grid impedance estimation", "estimating impedance of a power grid", "school enrollment"])
    assert vectors.shape == (3, 384) and embedder.max_tokens == 256
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]
    assert truncated_count(embedder, ["word " * 300, "short"]) == 1
