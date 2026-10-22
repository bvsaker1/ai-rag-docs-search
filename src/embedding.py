"""
Ingestion step 3: embed every chunk with one embedding model.

Vectors are L2-normalized float32, so inner product equals cosine similarity
(what the FAISS index searches with). Each model reads at most `max_tokens` of
its own tokens and silently drops the rest — all-MiniLM-L6-v2 reads 256,
all-mpnet-base-v2 384 — so chunks longer than that are counted and logged as
truncated.
"""

import functools
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from config import EMBEDDING_BATCH_SIZE, Logger
from models import Chunk


class Embedder(ABC):
    name: str

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of each vector."""

    @property
    def max_tokens(self) -> Optional[int]:
        """Tokens the model reads per text before truncating; None when unlimited or unknown."""
        return None

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """One L2-normalized float32 row per text."""

    def count_tokens(self, texts: list[str]) -> Optional[list[int]]:
        """Each text's length in the model's own tokens; None when unknown."""
        return None


class SentenceTransformerEmbedder(Embedder):
    """A sentence-transformers model run locally, loaded on first use."""

    def __init__(self, model_name: str, batch_size: int = EMBEDDING_BATCH_SIZE) -> None:
        self.name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.name)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    @property
    def max_tokens(self) -> Optional[int]:
        return self.model.max_seq_length

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False)
        return vectors.astype(np.float32)

    def count_tokens(self, texts: list[str]) -> Optional[list[int]]:
        encoded = self.model.tokenizer(texts, add_special_tokens=True, truncation=False, verbose=False)
        return [len(ids) for ids in encoded["input_ids"]]


EMBEDDERS: dict[str, type[Embedder]] = {"sentence-transformers": SentenceTransformerEmbedder}


@functools.cache
def make_embedder(model_name: str, backend: str = "sentence-transformers") -> Embedder:
    """One shared instance per model, so a model loads once per run (the semantic
    chunker's boundary model is usually also a retrieval model)."""
    if backend not in EMBEDDERS:
        raise ValueError(f"Unknown embedding backend {backend!r}; expected one of {', '.join(EMBEDDERS)}")
    return EMBEDDERS[backend](model_name)


def truncated_count(embedder: Embedder, texts: list[str]) -> Optional[int]:
    """How many texts exceed the model's token limit; None when that can't be known."""
    if embedder.max_tokens is None:
        return None
    counts = embedder.count_tokens(texts)
    return None if counts is None else sum(n > embedder.max_tokens for n in counts)


def embed_chunks(chunks: list[Chunk], embedder: Embedder, log: Optional[Logger] = None) -> np.ndarray:
    """Vectors for every chunk's content, in chunk order. Logs how many chunks the model truncates."""
    texts = [c.content for c in chunks]
    truncated = truncated_count(embedder, texts)
    if truncated and log:
        log.event("chunks_truncated", embedding_model=embedder.name, truncated=truncated, chunks=len(chunks),
                  max_tokens=embedder.max_tokens)
    return embedder.embed(texts)
