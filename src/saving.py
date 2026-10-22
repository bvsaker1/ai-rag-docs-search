"""
Output paths and every read/write of pipeline artifacts.

Output (data/ directory), all keyed by ITERATION:
    data/parsed_{iteration}.jsonl                          — one Document per paper
    data/chunks_{iteration}_{method}.jsonl                 — one Chunk per line
    data/chunking_config_{iteration}_{method}.json         — the ChunkingConfig used
    data/embeddings_{iteration}_{method}_{model}.npz       — cached chunk embeddings
    data/faiss_{iteration}_{method}_{model}.index          — FAISS index over those embeddings
    data/bm25_{iteration}_{method}.pkl                     — BM25 index over the chunk text
    data/pipeline_summary_{iteration}.json                 — per-task status of the last pipeline run
    data/queries/query-set_{iteration}.json                — the selected test queries and their ground truth
    data/answers/{iteration}/{generator}/answer-{config}.json — one AnswerFile per configuration, e.g.
                                                             answer-recursive-mpnet-hybrid.json, answer-fixed-bm25.json
    data/answers/{iteration}/judgments.jsonl               — cached LLM-judge scores (generator comparison)
    data/generator_comparison_{iteration}.json             — per-generator average judge scores
    data/metrics_{iteration}.jsonl                         — one EvaluationResult per configuration
    data/best_config_{iteration}.json                      — the best configuration (NDCG@K, then MRR)
    data/visualizations/{iteration}/*.png                  — evaluation charts

Logs (logs/ directory):
    logs/{LOG_BASE}_{iteration}.md                         — readable evaluation report
    logs/iteration_log.md, logs/iteration_log.jsonl        — change, reason, metrics before/after, delta

Each function takes an optional `data_dir` so tests can write to a temporary directory.
"""

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from pydantic import BaseModel

from config import DATA_DIR, LOG_BASE, LOGS_DIR, model_key
from models import Chunk, ChunkingConfig, RetrievalConfig


def _model_key(model: str) -> str:
    return model.replace("/", "-")


# ── Output Paths ──────────────────────────────────────────────────────────────

def parsed_path(iteration: int, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"parsed_{iteration}.jsonl"


def chunks_path(iteration: int, method: str, data_dir: Path = DATA_DIR) -> Path:
    """`method` is a file key: fixed / recursive / semantic."""
    return data_dir / f"chunks_{iteration}_{method}.jsonl"


def chunking_config_path(iteration: int, method: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"chunking_config_{iteration}_{method}.json"


def embeddings_path(iteration: int, method: str, model: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"embeddings_{iteration}_{method}_{_model_key(model)}.npz"


def faiss_index_path(iteration: int, method: str, model: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"faiss_{iteration}_{method}_{_model_key(model)}.index"


def bm25_index_path(iteration: int, method: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"bm25_{iteration}_{method}.pkl"


def pipeline_summary_path(iteration: int, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"pipeline_summary_{iteration}.json"


def query_set_path(iteration: int, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "queries" / f"query-set_{iteration}.json"


def answers_dir(iteration: int, generator: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "answers" / str(iteration) / model_key(generator)


def answer_path(iteration: int, generator: str, config: RetrievalConfig, data_dir: Path = DATA_DIR) -> Path:
    return answers_dir(iteration, generator, data_dir) / f"answer-{config.name}.json"


def judgments_path(iteration: int, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "answers" / str(iteration) / "judgments.jsonl"


def generator_comparison_path(iteration: int, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"generator_comparison_{iteration}.json"


def metrics_path(iteration: int, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"metrics_{iteration}.jsonl"


def best_config_path(iteration: int, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"best_config_{iteration}.json"


def visualizations_dir(iteration: int, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "visualizations" / str(iteration)


def markdown_log_path(iteration: int, logs_dir: Path = LOGS_DIR) -> Path:
    return logs_dir / f"{LOG_BASE}_{iteration}.md"


def iteration_log_path(logs_dir: Path = LOGS_DIR, suffix: str = ".md") -> Path:
    return logs_dir / f"iteration_log{suffix}"


# ── JSON / JSONL ──────────────────────────────────────────────────────────────

def write_jsonl(path: Path, records: Iterable[BaseModel]) -> int:
    """Overwrite `path` with one JSON-serialized model per line. Returns the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
            count += 1
    return count


def read_jsonl[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    """Read every non-blank line of `path` and validate it as `model`."""
    with open(path, encoding="utf-8") as f:
        return [model.model_validate_json(line) for line in f if line.strip()]


def save_chunking_config(path: Path, config: ChunkingConfig) -> None:
    path.write_text(config.model_dump_json(indent=2) + "\n")


def load_chunking_config(path: Path) -> ChunkingConfig:
    return ChunkingConfig.model_validate_json(path.read_text())


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def save_model(path: Path, record: BaseModel) -> None:
    """One model as indented JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2) + "\n")


def load_model[T: BaseModel](path: Path, model: type[T]) -> T:
    return model.model_validate_json(path.read_text())


# ── Embeddings ────────────────────────────────────────────────────────────────

def save_embeddings(path: Path, chunks: list[Chunk], vectors: np.ndarray) -> None:
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    np.savez(path, vectors=vectors, chunk_ids=np.array([str(c.id) for c in chunks]))


def load_embeddings(path: Path, chunks: Optional[list[Chunk]] = None) -> np.ndarray:
    """Cached vectors in chunk order. When `chunks` is given, fails if the cache is for different chunks."""
    with np.load(path) as data:
        if chunks is not None and list(data["chunk_ids"]) != [str(c.id) for c in chunks]:
            raise ValueError(f"{path.name} does not match the current chunks — re-run the embedding step")
        return data["vectors"]
