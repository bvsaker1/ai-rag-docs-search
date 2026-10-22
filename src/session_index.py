"""
Backend for the Streamlit app (app.py): an in-memory index of uploaded PDFs, and
retrievers that search the uploads, the ingested corpus, or both.

Uploads go through the same components as ingestion — parser, preprocessor, chunker,
embedder, BM25 and FAISS indexes — but nothing is written to disk, so the corpus
indexes the evaluation was scored on never change. "Both" fuses the corpus and the
uploads by reciprocal rank (MergedRetriever) before any reranking.
"""

import hashlib
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Optional

import embedding
from chunking import build_config, chunk_documents, make_chunker
from config import PDF_PARSE_LIB, method_key
from indexing import BM25Index, make_vector_index
from models import Chunk, Document, RetrievalConfig, RetrieverType
from parser import EmptyDocumentError, build_document, make_parser, make_preprocessor
from query_phase import QuerySettings
from retrieval import MergedRetriever, RerankedRetriever, Retriever, build_retriever, make_reranker, make_retriever
from saving import bm25_index_path, chunks_path, faiss_index_path, read_jsonl


class Scope(StrEnum):
    UPLOADS = "Uploaded files"
    CORPUS = "Corpus"
    BOTH = "Both"


@dataclass
class UploadedFile:
    name: str
    data: bytes

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.data).hexdigest()[:16]


@dataclass
class SessionIndex:
    """Chunks of the uploaded PDFs for one chunking method, with a BM25 index and one
    vector index per embedding model (built when first needed)."""
    chunk_method: str
    documents: list[Document]
    chunks: list[Chunk]
    errors: dict[str, str] = field(default_factory=dict)          # file name → why it was skipped
    bm25: Optional[BM25Index] = None
    vectors: dict[str, object] = field(default_factory=dict)       # embedding model → vector index

    def vector_index(self, model: str, vector_db: str = "FAISS"):
        if model not in self.vectors:
            embedder = embedding.make_embedder(model)
            index = make_vector_index(vector_db)
            index.build(self.chunks, embedding.embed_chunks(self.chunks, embedder))
            self.vectors[model] = index
        return self.vectors[model]

    def retriever(self, config: RetrievalConfig, alpha: float, candidates: int) -> Retriever:
        model = config.embedding_model
        needs_vectors = config.retrieval_method is not RetrieverType.BM25
        return build_retriever(config.retrieval_method, self.chunks, self.bm25,
                               self.vector_index(model) if needs_vectors else None,
                               embedding.make_embedder(model) if needs_vectors else None, alpha, candidates)


def build_session_index(files: list[UploadedFile], chunk_method: str, chunk_size: int, overlap: int,
                        semantic_model: str, parser_name: str = PDF_PARSE_LIB) -> SessionIndex:
    """Parse, clean, and chunk uploaded PDFs; files that can't be parsed are reported in `errors`."""
    parser, preprocessor = make_parser(parser_name), make_preprocessor("basic")
    documents, errors = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            path = Path(tmp) / Path(f.name).name
            path.write_bytes(f.data)
            try:
                documents.append(build_document(parser.parse(path), preprocessor))
            except EmptyDocumentError:
                errors[f.name] = "no extractable text (scanned or image-only PDF?)"
            except Exception as e:
                errors[f.name] = f"could not be parsed ({type(e).__name__}: {e})"
    config = build_config(chunk_method, chunk_size, overlap, parser=parser_name)
    boundary = embedding.make_embedder(semantic_model) if config.chunker == "semantic" else None
    chunks = chunk_documents(documents, make_chunker(config, boundary)) if documents else []
    index = SessionIndex(chunk_method=method_key(chunk_method), documents=documents, chunks=chunks, errors=errors)
    if chunks:
        index.bm25 = BM25Index()
        index.bm25.build(chunks)
    return index


def corpus_retriever(s: QuerySettings, config: RetrievalConfig) -> Retriever:
    """The first-stage retriever over the iteration's ingested corpus (no reranking)."""
    it, dd, key, model = s.iteration, s.data_dir, config.chunk_method, config.embedding_model
    path = chunks_path(it, key, dd)
    if not path.exists():
        raise FileNotFoundError(f"No {key} chunks for iteration {it} — run `pipeline.py ingest --iteration {it}`")
    return make_retriever(config.retrieval_method, read_jsonl(path, Chunk), bm25_path=bm25_index_path(it, key, dd),
                          faiss_path=faiss_index_path(it, key, model, dd) if model else None,
                          embedder=embedding.make_embedder(model) if model else None, alpha=s.hybrid_alpha,
                          candidates=s.hybrid_candidates)


def scoped_retriever(scope: Scope, config: RetrievalConfig, s: QuerySettings,
                     session: Optional[SessionIndex] = None, corpus: Optional[Retriever] = None) -> Retriever:
    """The retriever for a search scope, reranked when config.reranker is set. `corpus` is a
    first-stage corpus retriever (built with corpus_retriever when omitted)."""
    parts: list[Retriever] = []
    if scope in (Scope.CORPUS, Scope.BOTH):
        parts.append(corpus or corpus_retriever(s, config))
    if scope in (Scope.UPLOADS, Scope.BOTH):
        if session is None or not session.chunks:
            if scope is Scope.UPLOADS:
                raise ValueError("Upload at least one PDF with extractable text to search uploaded files")
        else:
            parts.append(session.retriever(config, s.hybrid_alpha, s.hybrid_candidates))
    retriever = parts[0] if len(parts) == 1 else MergedRetriever(parts)
    reranker = make_reranker(config.reranker)
    return RerankedRetriever(retriever, reranker, s.rerank_candidates) if reranker else retriever
