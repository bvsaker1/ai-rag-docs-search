"""
Ingestion step 4: build and persist the search indexes retrievers will query.

    FaissIndex  exact inner-product search over L2-normalized chunk embeddings
                (= cosine similarity); one per chunking method × embedding model
    BM25Index   Okapi BM25 over the chunk text; one per chunking method

Hybrid retrieval has no index of its own: it fuses BM25 and FAISS rankings at
query time. Index positions follow the chunk file's order, so position i is the
i-th chunk in data/chunks_{iteration}_{method}.jsonl.
"""

import pickle
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from models import Chunk

# Words, plus numbers kept whole with their separators ("7,274", "2.5", "fy2010").
_TOKEN_RE = re.compile(r"\w+(?:[.,']\w+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _faiss():
    """faiss, imported after config, which lets faiss's and torch's OpenMP runtimes coexist."""
    import config  # noqa: F401
    import faiss
    return faiss


class Index(ABC):
    name: str

    @abstractmethod
    def build(self, chunks: list[Chunk], vectors: Optional[np.ndarray] = None) -> None:
        """Index `chunks` (and their `vectors`, for vector indexes)."""

    @abstractmethod
    def search(self, query: str | np.ndarray, k: int) -> list[tuple[int, float]]:
        """(chunk position, score) of the top `k`, best first. Query text for keyword
        indexes, a normalized query vector for vector indexes."""

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Index": ...

    @abstractmethod
    def __len__(self) -> int: ...


class FaissIndex(Index):
    name = "faiss"

    def __init__(self) -> None:
        self._index = None

    def build(self, chunks: list[Chunk], vectors: Optional[np.ndarray] = None) -> None:
        faiss = _faiss()
        if vectors is None or len(vectors) != len(chunks):
            raise ValueError("FaissIndex needs one vector per chunk")
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, query: str | np.ndarray, k: int) -> list[tuple[int, float]]:
        if isinstance(query, str):
            raise TypeError("FaissIndex searches with a query vector; embed the query first")
        scores, ids = self._index.search(np.asarray(query, dtype=np.float32).reshape(1, -1), min(k, len(self)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def save(self, path: Path) -> None:
        _faiss().write_index(self._index, str(path))

    @classmethod
    def load(cls, path: Path) -> "FaissIndex":
        index = cls()
        index._index = _faiss().read_index(str(path))
        return index

    def __len__(self) -> int:
        return 0 if self._index is None else self._index.ntotal


class BM25Index(Index):
    name = "bm25"

    def __init__(self) -> None:
        self._bm25 = None
        self._count = 0

    def build(self, chunks: list[Chunk], vectors: Optional[np.ndarray] = None) -> None:
        from rank_bm25 import BM25Okapi

        if not chunks:
            raise ValueError("BM25Index needs at least one chunk")
        self._bm25 = BM25Okapi([tokenize(c.content) for c in chunks])
        self._count = len(chunks)

    def search(self, query: str | np.ndarray, k: int) -> list[tuple[int, float]]:
        if not isinstance(query, str):
            raise TypeError("BM25Index searches with query text")
        scores = self._bm25.get_scores(tokenize(query))
        top = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in top]

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "count": self._count}, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        index = cls()
        index._bm25, index._count = data["bm25"], data["count"]
        return index

    def __len__(self) -> int:
        return self._count


# VECTOR_DB names → vector index classes.
VECTOR_INDEXES: dict[str, type[Index]] = {"FAISS": FaissIndex}


def make_vector_index(name: str) -> Index:
    cls = VECTOR_INDEXES.get(name.upper())
    if cls is None:
        raise ValueError(f"Unknown vector DB {name!r}; expected one of {', '.join(VECTOR_INDEXES)}")
    return cls()


def build_index(index: Index, chunks: list[Chunk], path: Path, vectors: Optional[np.ndarray] = None) -> Index:
    """Build `index` over `chunks` and save it to `path`."""
    index.build(chunks, vectors)
    index.save(path)
    return index
