"""
Pydantic models for the RAG pipeline: the five models from INSTRUCTIONS.md
(Document, Chunk, RetrievalResult, QAResponse, Citation) plus the configuration,
section, query-set, answer-file, and judging models.

Validation happens on construction and on every read from disk, so malformed
parser, chunker, or LLM output fails loudly at the step that produced it. The LLM's
own structured output (generation.GeneratedAnswer, JudgeScores) is validated by
Instructor, which re-asks the model when it doesn't fit.
"""

import uuid
from enum import StrEnum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

# Every id is a uuid5 in this namespace, so re-running ingestion yields the same ids.
ID_NAMESPACE = uuid.NAMESPACE_URL

# Pages are joined with this to form Document.content.
PAGE_SEPARATOR = "\n\n"


def _check_span(start_char: int, end_char: int) -> None:
    if end_char <= start_char:
        raise ValueError("end_char must be after start_char")


# ── Documents ─────────────────────────────────────────────────────────────────

class Section(BaseModel):
    """A literal section of a paper. start_char/end_char are offsets into
    Document.content; the first section starts at 0 and each ends where the next begins."""
    index: int = Field(ge=0)
    title: str
    level: int = Field(ge=1)
    page_number: int = Field(ge=1)
    start_char: int = Field(ge=0)
    end_char: int

    @model_validator(mode="after")
    def _end_after_start(self) -> "Section":
        _check_span(self.start_char, self.end_char)
        return self


class DocumentMetadata(BaseModel):
    """`section_source` says how sections were found: "toc" (the PDF's bookmark
    outline), "font" (lines set larger or bold-numbered), or "page" (no headings
    found: one section per page). page_starts[i] is where page i+1 starts in content."""
    source: str                     # file name, e.g. 2410.14077v2.pdf
    arxiv_id: str
    title: Optional[str] = None
    author: Optional[str] = None
    page_count: int = Field(ge=1)
    parser: str
    has_qrels: bool = False
    page_starts: list[int]
    section_source: str
    sections: list[Section] = Field(default_factory=list)


class Document(BaseModel):
    id: UUID
    content: str = Field(min_length=1)
    metadata: DocumentMetadata

    @model_validator(mode="after")
    def _pages_and_sections_fit(self) -> "Document":
        starts = self.metadata.page_starts
        if len(starts) != self.metadata.page_count or starts != sorted(starts) or (starts and starts[0] != 0):
            raise ValueError("page_starts must hold one ascending offset per page, starting at 0")
        if any(s.end_char > len(self.content) for s in self.metadata.sections):
            raise ValueError("a section ends past the end of content")
        return self

    @staticmethod
    def make_id(source: str) -> UUID:
        return uuid.uuid5(ID_NAMESPACE, f"document/{source}")


# ── Chunking ──────────────────────────────────────────────────────────────────

class ChunkingConfig(BaseModel):
    """chunk_size is the window (fixed) or maximum (recursive, semantic) in tokens of
    `tokenizer`; overlap is in tokens. Method-specific settings are None when unused.
    min_tokens: recursive — sections under it merge into the next; semantic — the
    least a chunk holds before it may be cut at a breakpoint."""
    parser: str
    chunker: str
    chunk_size: int = Field(gt=0)
    overlap: int = Field(ge=0)
    tokenizer: str
    separators: Optional[list[str]] = None              # recursive, in splitting order
    boundary_embedding_model: Optional[str] = None      # semantic
    breakpoint_percentile: Optional[float] = None       # semantic
    min_tokens: Optional[int] = None                    # recursive, semantic

    @model_validator(mode="after")
    def _overlap_below_size(self) -> "ChunkingConfig":
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        return self

    @property
    def name(self) -> str:
        """e.g. pymupdf_fixed_token_chunk_size512_overlap75"""
        return f"{self.parser}_{self.chunker}_chunk_size{self.chunk_size}_overlap{self.overlap}"


class ChunkMetadata(BaseModel):
    """page_number is where the chunk starts, end_page_number where it ends.
    start_char/end_char are offsets into the document's content. The section is
    the one the chunk starts in. token_count is the tokens the chunker packed
    (re-tokenizing the content alone can differ by a token or two at the edges)."""
    document_id: UUID
    source: str
    arxiv_id: str
    page_number: int = Field(ge=1)
    end_page_number: int = Field(ge=1)
    start_char: int = Field(ge=0)
    end_char: int
    chunk_index: int = Field(ge=0)
    section_index: Optional[int] = None
    section_title: Optional[str] = None
    method: str
    token_count: int = Field(ge=1)

    @model_validator(mode="after")
    def _spans_ordered(self) -> "ChunkMetadata":
        _check_span(self.start_char, self.end_char)
        if self.end_page_number < self.page_number:
            raise ValueError("end_page_number must not be before page_number")
        return self


class Chunk(BaseModel):
    """`embedding` stays None in the chunk files; vectors are cached separately
    (see saving.save_embeddings) so one chunk file serves every embedding model."""
    id: UUID
    content: str = Field(min_length=1)
    metadata: ChunkMetadata
    embedding: Optional[list[float]] = None

    @model_validator(mode="after")
    def _content_matches_span(self) -> "Chunk":
        if len(self.content) != self.metadata.end_char - self.metadata.start_char:
            raise ValueError("content length must equal end_char - start_char")
        return self

    @staticmethod
    def make_id(config: ChunkingConfig, source: str, chunk_index: int) -> UUID:
        return uuid.uuid5(ID_NAMESPACE, f"{config.name}/{source}/chunk{chunk_index}")


# ── Retrieval & Generation (query phase) ──────────────────────────────────────

class RetrieverType(StrEnum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float
    retriever_type: RetrieverType


class Citation(BaseModel):
    chunk_id: UUID
    source: str = Field(min_length=1)   # file name
    page_number: int = Field(ge=1)
    text_snippet: str
    relevance_score: float


class QAResponse(BaseModel):
    query: str = Field(min_length=1)
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    chunks_used: list[Chunk] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# ── Query set ─────────────────────────────────────────────────────────────────

class QueryRecord(BaseModel):
    """One benchmark query with its ground truth: the qrels paper and section, and
    the reference answer from answers.json."""
    query_id: str
    query: str = Field(min_length=1)
    type: str                       # abstractive | extractive
    source: str                     # text | text-image | text-table | text-table-image
    doc_id: str
    section_id: int
    reference_answer: str


class QuerySet(BaseModel):
    seed: int
    count: int
    abstractive: int
    source_mix: dict[str, float]
    queries: list[QueryRecord]


# ── Answers ───────────────────────────────────────────────────────────────────

class RetrievalConfig(BaseModel):
    """One of the chunking × embedding × retrieval configurations. embedding_model is
    None for BM25, which uses no embeddings."""
    chunk_method: str                    # file key: fixed | recursive | semantic
    embedding_model: Optional[str] = None
    retrieval_method: RetrieverType
    top_k: int = Field(gt=0)
    reranker: str = "none"
    hybrid_alpha: Optional[float] = None

    @property
    def name(self) -> str:
        """e.g. recursive-mpnet-hybrid, fixed-bm25"""
        from config import model_key
        parts = [self.chunk_method, model_key(self.embedding_model) if self.embedding_model else None,
                 self.retrieval_method.value, None if self.reranker == "none" else f"rerank-{self.reranker}"]
        return "-".join(p for p in parts if p)


class RetrievedRef(BaseModel):
    """A retrieved chunk by reference (the chunk itself is in QAResponse.chunks_used)."""
    rank: int = Field(ge=1)
    chunk_id: UUID
    score: float
    retriever_type: RetrieverType
    source: str
    page_number: int = Field(ge=1)
    section_title: Optional[str] = None


class AnswerEntry(BaseModel):
    query_id: str
    query: str
    retrieved: list[RetrievedRef] = Field(default_factory=list)
    response: Optional[QAResponse] = None
    error: Optional[str] = None
    retrieval_seconds: float = 0.0
    generation_seconds: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class AnswerFile(BaseModel):
    """Every query's answer for one configuration and generator."""
    iteration: int
    config: RetrievalConfig
    generator: str
    created_at: str
    entries: list[AnswerEntry]


# ── Judging ───────────────────────────────────────────────────────────────────

class JudgeScores(BaseModel):
    """LLM-as-judge scores on a 1-5 scale."""
    relevance: int = Field(ge=1, le=5, description="Does the answer address the question asked?")
    accuracy: int = Field(ge=1, le=5, description="Is it factually consistent with the reference answer and sources?")
    completeness: int = Field(ge=1, le=5, description="Does it cover everything the reference answer covers?")
    citation_quality: int = Field(ge=1, le=5, description="Are claims attributed to sources that support them?")
    reasoning: str = Field(description="One or two sentences justifying the scores.")

    @property
    def overall(self) -> float:
        return (self.relevance + self.accuracy + self.completeness + self.citation_quality) / 4


class Judgment(BaseModel):
    """One LLM-judge verdict on one answer; `answer_hash` identifies which answer, so a
    regenerated answer is judged again."""
    generator: str
    configuration: str
    query_id: str
    judge: str
    answer_hash: Optional[str] = None
    scores: Optional[JudgeScores] = None
    error: Optional[str] = None

    @property
    def key(self) -> tuple[str, str, str, str, Optional[str]]:
        return self.generator, self.configuration, self.query_id, self.judge, self.answer_hash


# ── Evaluation ────────────────────────────────────────────────────────────────

class QueryMetrics(BaseModel):
    """Retrieval metrics at K for one query and configuration.

    Section level (primary): relevant chunks are those overlapping the qrels section.
    max_precision is the best Precision@K possible (min(K, relevant) / K).
    Paper level (secondary): every chunk of the qrels paper counts as relevant."""
    query_id: str
    type: str
    source: str
    relevant_chunks: int
    recall: float
    precision: float
    mrr: float
    ndcg: float
    hit: float
    max_precision: float
    paper_precision: float
    paper_mrr: float
    paper_ndcg: float
    paper_hit: float
    retrieval_ms: Optional[float] = None
    generation_seconds: Optional[float] = None
    judge: Optional[JudgeScores] = None


class EvaluationResult(BaseModel):
    """One configuration's evaluation: means over the query set plus every query's metrics."""
    experiment_id: str
    iteration: int
    config: RetrievalConfig
    chunking: dict
    generator: str
    judge_model: Optional[str] = None
    k: int
    metrics: dict[str, float]
    judge_scores: dict[str, float] = Field(default_factory=dict)
    latency: dict[str, float] = Field(default_factory=dict)
    per_query: list[QueryMetrics]

