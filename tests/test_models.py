import uuid

import pytest
from pydantic import ValidationError

from conftest import make_document
from models import (
    Chunk, ChunkingConfig, ChunkMetadata, Citation, Document, DocumentMetadata, QAResponse, RetrievalResult,
    RetrieverType, Section,
)

DOC_ID = uuid.uuid4()


def chunk_metadata(**overrides) -> dict:
    fields = dict(document_id=DOC_ID, source="a.pdf", arxiv_id="a", page_number=1, end_page_number=1,
                  start_char=0, end_char=5, chunk_index=0, method="fixed_token", token_count=2)
    return {**fields, **overrides}


def make_chunk(content: str = "hello", **meta) -> Chunk:
    return Chunk(id=uuid.uuid4(), content=content, metadata=ChunkMetadata(**chunk_metadata(**meta)))


def test_chunk_requires_content_to_match_its_span():
    assert make_chunk("hello").metadata.end_char == 5
    with pytest.raises(ValidationError, match="content length"):
        make_chunk("hello!")


def test_chunk_metadata_rejects_reversed_spans_and_pages():
    with pytest.raises(ValidationError, match="end_char"):
        ChunkMetadata(**chunk_metadata(start_char=5, end_char=5))
    with pytest.raises(ValidationError, match="end_page_number"):
        ChunkMetadata(**chunk_metadata(page_number=3, end_page_number=2))


def test_chunk_rejects_empty_content_and_non_uuid_ids():
    with pytest.raises(ValidationError):
        Chunk(id=uuid.uuid4(), content="", metadata=ChunkMetadata(**chunk_metadata(end_char=1)))
    with pytest.raises(ValidationError):
        Chunk(id="not-a-uuid", content="hello", metadata=ChunkMetadata(**chunk_metadata()))


def test_chunk_round_trips_through_json_with_optional_embedding():
    chunk = make_chunk().model_copy(update={"embedding": [0.1, 0.2]})
    assert Chunk.model_validate_json(chunk.model_dump_json()) == chunk


def test_document_validates_page_starts_and_section_bounds():
    meta = dict(source="a.pdf", arxiv_id="a", page_count=2, parser="pymupdf", section_source="page")
    Document(id=DOC_ID, content="page one\n\npage two", metadata=DocumentMetadata(**meta, page_starts=[0, 10]))
    with pytest.raises(ValidationError, match="page_starts"):
        Document(id=DOC_ID, content="x", metadata=DocumentMetadata(**meta, page_starts=[0]))
    section = Section(index=0, title="All", level=1, page_number=1, start_char=0, end_char=99)
    with pytest.raises(ValidationError, match="past the end"):
        Document(id=DOC_ID, content="short", metadata=DocumentMetadata(**meta, page_starts=[0, 1], sections=[section]))


def test_ids_are_deterministic():
    config = ChunkingConfig(parser="pymupdf", chunker="fixed_token", chunk_size=512, overlap=75, tokenizer="cl100k_base")
    assert Document.make_id("a.pdf") == Document.make_id("a.pdf") != Document.make_id("b.pdf")
    assert Chunk.make_id(config, "a.pdf", 0) == Chunk.make_id(config, "a.pdf", 0)
    assert Chunk.make_id(config, "a.pdf", 0) != Chunk.make_id(config, "a.pdf", 1)
    assert make_document(["one"]).id == make_document(["two"]).id     # same source, same id


def test_chunking_config_requires_overlap_below_size_and_names_itself():
    config = ChunkingConfig(parser="pymupdf", chunker="recursive", chunk_size=512, overlap=75, tokenizer="cl100k_base")
    assert config.name == "pymupdf_recursive_chunk_size512_overlap75"
    with pytest.raises(ValidationError, match="overlap"):
        ChunkingConfig(parser="pymupdf", chunker="recursive", chunk_size=10, overlap=10, tokenizer="cl100k_base")


def test_retrieval_result_accepts_only_known_retriever_types():
    result = RetrievalResult(chunk=make_chunk(), score=0.5, retriever_type="bm25")
    assert result.retriever_type is RetrieverType.BM25
    with pytest.raises(ValidationError):
        RetrievalResult(chunk=make_chunk(), score=0.5, retriever_type="tfidf")


def test_qa_response_bounds_confidence_and_validates_citations():
    citation = Citation(chunk_id=uuid.uuid4(), source="a.pdf", page_number=1, text_snippet="x", relevance_score=0.9)
    response = QAResponse(query="q", answer="a", citations=[citation], chunks_used=[make_chunk()], confidence=0.8)
    assert response.citations[0].page_number == 1
    with pytest.raises(ValidationError):
        QAResponse(query="q", answer="a", confidence=1.5)
    with pytest.raises(ValidationError):
        Citation(chunk_id=uuid.uuid4(), source="a.pdf", page_number=0, text_snippet="x", relevance_score=0.9)
