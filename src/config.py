"""
Shared configuration and the structured logger for the RAG pipeline.

Every module imports its settings from here so .env is read in exactly one place.
Output paths and file helpers live in saving.py.

Logs (logs/ directory):
    logs/{LOG_BASE}_{iteration}.jsonl  — structured event log
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# faiss-cpu and torch each bundle their own OpenMP runtime on macOS; whichever starts second aborts
# the process ("OMP: Error #15") unless this is set. OpenMP calls it unsupported, so
# test_integration checks FAISS search results against a numpy brute-force search.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ── Config ────────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

# Papers: the first MAX_PAPERS PDFs (sorted by arXiv id) that have queries in QRELS_FILE, plus the
# first MAX_NEGATIVES that don't (hard negatives: never the answer to a query).
PAPERS_DIR: str = os.getenv("PAPERS_DIR", "data/papers")
QRELS_FILE: str = os.getenv("QRELS_FILE", "data/pdf/arxiv/qrels.json")
QUERIES_FILE: str = os.getenv("QUERIES_FILE", "data/pdf/arxiv/queries.json")
ANSWERS_FILE: str = os.getenv("ANSWERS_FILE", "data/pdf/arxiv/answers.json")
CORPUS_DIR: str = os.getenv("CORPUS_DIR", "data/pdf/arxiv/corpus")
MAX_PAPERS: int = int(os.getenv("MAX_PAPERS", "50"))
MAX_NEGATIVES: int = int(os.getenv("MAX_NEGATIVES", "0"))
PDF_PARSE_LIB: str = os.getenv("PDF_PARSE_LIB", "pymupdf")

# CHUNK_SIZE and OVERLAP are measured in TIKTOKEN_ENCODING tokens. CHUNK_SIZE is the maximum for
# RECURSIVE and SEMANTIC.
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
OVERLAP: int = int(os.getenv("OVERLAP", "75"))
TIKTOKEN_ENCODING: str = os.getenv("TIKTOKEN_ENCODING", "cl100k_base")
CHUNK_METHOD_1: str = os.getenv("CHUNK_METHOD_1", "FIXED")
CHUNK_METHOD_2: str = os.getenv("CHUNK_METHOD_2", "RECURSIVE")
CHUNK_METHOD_3: str = os.getenv("CHUNK_METHOD_3", "SEMANTIC")
CHUNK_METHODS: list[str] = [CHUNK_METHOD_1, CHUNK_METHOD_2, CHUNK_METHOD_3]

# RECURSIVE: sections split by paragraph → sentence → word → token, packed to CHUNK_SIZE with OVERLAP
# tokens of overlap; chunks never cross sections. Sections under this many tokens merge into the next.
RECURSIVE_MIN_SECTION_TOKENS: int = int(os.getenv("RECURSIVE_MIN_SECTION_TOKENS", "50"))
# SEMANTIC: sentence embeddings decide cut points; independent of the retrieval embedding models.
SEMANTIC_CHUNK_EMBEDDING_MODEL: str = os.getenv("SEMANTIC_CHUNK_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
SEMANTIC_BREAKPOINT_PERCENTILE: float = float(os.getenv("SEMANTIC_BREAKPOINT_PERCENTILE", "95"))
SEMANTIC_MIN_TOKENS: int = int(os.getenv("SEMANTIC_MIN_TOKENS", "50"))

# sentence-transformers models, run locally.
EMBEDDING_MODEL_1: str = os.getenv("EMBEDDING_MODEL_1", "all-MiniLM-L6-v2")
EMBEDDING_MODEL_2: str = os.getenv("EMBEDDING_MODEL_2", "all-mpnet-base-v2")
EMBEDDING_MODELS: list[str] = [EMBEDDING_MODEL_1, EMBEDDING_MODEL_2]
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
VECTOR_DB: str = os.getenv("VECTOR_DB", "FAISS")

# Retrieval method names match RetrieverType: bm25 | dense | hybrid.
RETRIEVAL_1: str = os.getenv("RETRIEVAL_1", "bm25")
RETRIEVAL_2: str = os.getenv("RETRIEVAL_2", "dense")
RETRIEVAL_3: str = os.getenv("RETRIEVAL_3", "hybrid")
RETRIEVAL_METHODS: list[str] = [RETRIEVAL_1, RETRIEVAL_2, RETRIEVAL_3]
TOP_K: int = int(os.getenv("TOP_K", "5"))
# Hybrid: min-max normalized scores combined as HYBRID_ALPHA · dense + (1 - HYBRID_ALPHA) · bm25, over the
# top HYBRID_CANDIDATES from each.
HYBRID_ALPHA: float = float(os.getenv("HYBRID_ALPHA", "0.5"))
HYBRID_CANDIDATES: int = int(os.getenv("HYBRID_CANDIDATES", "50"))

# Reranking: none | cross-encoder (local CROSS_ENCODER_MODEL) | cohere (COHERE_RERANK_MODEL via OpenRouter).
# When on, the first stage's top RERANK_CANDIDATES are reordered and cut to TOP_K.
RERANKER: str = os.getenv("RERANKER", "none")
RERANK_CANDIDATES: int = int(os.getenv("RERANK_CANDIDATES", "20"))
CROSS_ENCODER_MODEL: str = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
COHERE_RERANK_MODEL: str = os.getenv("COHERE_RERANK_MODEL", "cohere/rerank-v3.5")

# ── Query set ──────────────────────────────────────────────────────────────────
# QUERY_COUNT queries from the ingested papers (at most one per paper): QUERY_ABSTRACTIVE abstractive, the
# rest extractive, split across sources by QUERY_SOURCE_MIX (largest-remainder rounding).
QUERY_COUNT: int = int(os.getenv("QUERY_COUNT", "12"))
QUERY_ABSTRACTIVE: int = int(os.getenv("QUERY_ABSTRACTIVE", "8"))
QUERY_SOURCE_MIX: str = os.getenv("QUERY_SOURCE_MIX", "text:0.65,text-image:0.25,text-table:0.05,text-table-image:0.05")
QUERY_SEED: int = int(os.getenv("QUERY_SEED", "42"))

# ── Generation & judging (OpenRouter) ──────────────────────────────────────────
GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "openai/gpt-6-luna")
# Models compared by compare_generators.py, scored by JUDGE_MODEL (from a different provider).
GENERATION_CANDIDATES: list[str] = [m.strip() for m in os.getenv(
    "GENERATION_CANDIDATES", "anthropic/claude-haiku-5.5,openai/gpt-6-luna,google/gemini-2.5-flash-lite").split(",")
    if m.strip()]
GENERATION_TEMPERATURE: float = float(os.getenv("GENERATION_TEMPERATURE", "0"))
JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "deepseek/deepseek-v4-flash")
JUDGE_TEMPERATURE: float = float(os.getenv("JUDGE_TEMPERATURE", "0"))
LLM_CONCURRENCY: int = int(os.getenv("LLM_CONCURRENCY", "8"))

# ── Evaluation ─────────────────────────────────────────────────────────────────
# A chunk is relevant to a query when at least RELEVANCE_MIN_OVERLAP of its word RELEVANCE_NGRAM-grams
# appear in the qrels section's corpus text, or of the section's appear in the chunk.
RELEVANCE_MIN_OVERLAP: float = float(os.getenv("RELEVANCE_MIN_OVERLAP", "0.5"))
RELEVANCE_NGRAM: int = int(os.getenv("RELEVANCE_NGRAM", "3"))
METRIC_K: int = int(os.getenv("METRIC_K", "5"))
LLM_MAX_ATTEMPTS: int = int(os.getenv("LLM_MAX_ATTEMPTS", "3"))
# Per-call network timeout (seconds); openai-python defaults to 600s.
LLM_REQUEST_TIMEOUT_SECONDS: float = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "60"))

LOG_BASE: str = os.getenv("LOG_BASE", "rag_log")
ITERATION: int = int(os.getenv("ITERATION", "1"))

# .env chunking method names → the method names stored on Chunk / ChunkingConfig.
CHUNK_METHOD_NAMES: dict[str, str] = {
    "FIXED": "fixed_token",
    "RECURSIVE": "recursive",
    "SEMANTIC": "semantic",
}

# Resolve relative to project root (one level above src/)
PROJECT_ROOT: Path = Path(__file__).parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


def method_key(method: str) -> str:
    """File-name key for a .env chunking method: FIXED → "fixed"."""
    return method.lower()


def model_key(model: str) -> str:
    """Short file-name key for a model: all-MiniLM-L6-v2 → "minilm", all-mpnet-base-v2 → "mpnet",
    anthropic/claude-haiku-5.5 → "claude-haiku-5.5"."""
    name = model.split("/")[-1].lower()
    if name.startswith("all-"):
        return name.removeprefix("all-").split("-")[0]
    return name


def resolve_project_path(path: str) -> Path:
    """Resolve a .env path (e.g. PAPERS_DIR) relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def log_path(iteration: int) -> Path:
    return LOGS_DIR / f"{LOG_BASE}_{iteration}.jsonl"


# ── Logger ────────────────────────────────────────────────────────────────────

class Logger:
    """Appends structured JSONL events to logs/{LOG_BASE}_{iteration}.jsonl.

    Every record carries the step name, so one log file holds the whole
    pipeline run for an iteration. `path` overrides the log file (tests)."""

    def __init__(self, iteration: int, step: str, path: Optional[Path] = None, **context) -> None:
        self._path = path or log_path(iteration)
        self._step = step
        self._context = context
        self._file = open(self._path, "a", encoding="utf-8")

    def bind(self, **context) -> "Logger":
        """Fields added to every following event, e.g. bind(chunk_method=...).
        Passing None for a field removes it."""
        self._context = {k: v for k, v in {**self._context, **context}.items() if v is not None}
        return self

    def event(self, event: str, **fields) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "step": self._step,
            "event": event,
            **self._context,
            **fields,
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


# ── LLM Client ────────────────────────────────────────────────────────────────

def make_openai_client():
    """A plain OpenAI SDK client pointed at OpenRouter."""
    from openai import OpenAI

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to your environment or .env file.")
    return OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL, timeout=LLM_REQUEST_TIMEOUT_SECONDS)


def make_client():
    """An instructor client that routes every model through OpenRouter. Tools mode: the
    provider's function calling builds the JSON, which every candidate model supports
    (JSON mode is enforced by OpenAI but only requested from Anthropic, so Claude often
    answered in prose first and needed a re-ask; Gemini wrote invalid JSON escapes)."""
    import instructor

    return instructor.from_openai(make_openai_client(), mode=instructor.Mode.TOOLS)
