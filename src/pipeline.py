#!/usr/bin/env python3
"""
Pipeline: the three RAG phases — ingest, query, evaluate. (compare_generators.py
compares generator LLMs with the same judge.)

INGESTION
    parsing (1×) → chunking (per chunking method) → embedding (per method × embedding model)
      → indexing (FAISS per method × model, BM25 per method)

QUERY       query set → for each of the 15 configurations: retrieve top-K → rerank (if RERANKER is set)
              → generate answer → extract citations (see query_phase.py)
EVALUATION  saved answers → Recall/Precision/MRR/NDCG@K against corpus-section ground truth, warm latency,
              LLM-judge scores → metrics, best configuration, report, charts (see evaluate_phase.py)

Every component is chosen by name from .env (or the flags below) and built by its
module's factory, so any chunking × embedding × retrieval combination runs
without code changes.

An ingestion task is skipped when its outputs already exist, unless --force is
given, the task's step is the --only step, or a task it depends on ran in this
run (re-chunking RECURSIVE re-embeds and re-indexes only the recursive chunks).
A failed task is recorded and its dependents are skipped; the rest still run.

Usage:
    python pipeline.py ingest                       # run whatever isn't done yet
    python pipeline.py ingest --force               # rerun every task
    python pipeline.py ingest --only chunking       # rerun one step (all its tasks)
    python pipeline.py ingest --max-papers 5 --methods FIXED --models all-MiniLM-L6-v2
    python pipeline.py query                        # answer the query set with every configuration
    python pipeline.py query --generator openai/gpt-6-luna --reranker cross-encoder
    python pipeline.py query "What is VDF-RLS?" --config recursive-mpnet-hybrid
    python pipeline.py evaluate                     # score the iteration's answers, write metrics and charts
    python pipeline.py evaluate --iteration 2 --compare-to 1 --change "CHUNK_SIZE 512 → 256" --reason "..."

Output:
    data/pipeline_summary_{iteration}.json (plus each step's outputs; see saving.py)
"""

import argparse
import functools
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from config import (
    CHUNK_METHODS, CHUNK_SIZE, DATA_DIR, EMBEDDING_MODELS, GENERATION_MODEL, ITERATION, JUDGE_MODEL, MAX_NEGATIVES, MAX_PAPERS, OVERLAP,
    PAPERS_DIR, PDF_PARSE_LIB, QRELS_FILE, RERANKER, SEMANTIC_CHUNK_EMBEDDING_MODEL, VECTOR_DB, Logger, method_key,
    resolve_project_path,
)
from models import Chunk, Document
from saving import (
    bm25_index_path, chunking_config_path, chunks_path, embeddings_path, faiss_index_path, load_embeddings,
    parsed_path, pipeline_summary_path, read_jsonl, save_chunking_config, save_embeddings, save_json, write_jsonl,
)

# Flush every print so progress shows up as it happens.
print = functools.partial(print, flush=True)

STEP_NAMES = ["parsing", "chunking", "embedding", "indexing"]
PREPROCESSOR = "basic"


@dataclass
class IngestSettings:
    iteration: int = ITERATION
    papers_dir: Path = field(default_factory=lambda: resolve_project_path(PAPERS_DIR))
    qrels_path: Path = field(default_factory=lambda: resolve_project_path(QRELS_FILE))
    max_papers: int = MAX_PAPERS
    max_negatives: int = MAX_NEGATIVES
    parser: str = PDF_PARSE_LIB
    preprocessor: str = PREPROCESSOR
    methods: list[str] = field(default_factory=lambda: list(CHUNK_METHODS))
    models: list[str] = field(default_factory=lambda: list(EMBEDDING_MODELS))
    chunk_size: int = CHUNK_SIZE
    overlap: int = OVERLAP
    semantic_model: str = SEMANTIC_CHUNK_EMBEDDING_MODEL
    vector_db: str = VECTOR_DB
    force: bool = False
    only: Optional[str] = None
    data_dir: Path = DATA_DIR
    log_path: Optional[Path] = None     # default: logs/{LOG_BASE}_{iteration}.jsonl


@dataclass
class TaskResult:
    step: str
    keys: dict
    status: str                         # ran | skipped | failed | blocked (an input task failed)
    outputs: list[str]
    duration_seconds: float = 0.0
    error: Optional[str] = None

    @property
    def label(self) -> str:
        return " / ".join([self.step, *self.keys.values()])


class Runner:
    """Runs ingestion tasks in dependency order, deciding which to skip."""

    def __init__(self, settings: IngestSettings) -> None:
        self.s = settings
        self.results: list[TaskResult] = []

    def log(self, step: str, **context) -> Logger:
        return Logger(self.s.iteration, step, path=self.s.log_path, **context)

    def run(self, step: str, keys: dict, outputs: list[Path], work: Callable[[Logger], None],
            upstream: list[TaskResult]) -> TaskResult:
        """Run `work` unless it can be skipped. Returns the task's result."""
        s = self.s
        if any(u.status in ("failed", "blocked") for u in upstream):
            result = TaskResult(step, keys, "blocked", [str(p) for p in outputs])
        elif s.only is not None and s.only != step:
            result = TaskResult(step, keys, "skipped", [str(p) for p in outputs])
        elif not (s.force or s.only == step or any(u.status == "ran" for u in upstream)
                  or not all(p.exists() for p in outputs)):
            result = TaskResult(step, keys, "skipped", [str(p) for p in outputs])
        else:
            log = self.log(step, **keys)
            start = time.monotonic()
            try:
                work(log)
                result = TaskResult(step, keys, "ran", [str(p) for p in outputs],
                                    round(time.monotonic() - start, 2))
            except Exception as e:
                log.event("step_failed", error=f"{type(e).__name__}: {e}")
                result = TaskResult(step, keys, "failed", [str(p) for p in outputs],
                                    round(time.monotonic() - start, 2), f"{type(e).__name__}: {e}")
            finally:
                log.close()
        print(f"[{result.label}] {result.status}" + (f" ({result.duration_seconds}s)" if result.status == "ran" else "")
              + (f": {result.error}" if result.error else ""))
        self.results.append(result)
        return result


# ── Steps ─────────────────────────────────────────────────────────────────────

def parse_step(s: IngestSettings, out_path: Path, log: Logger) -> None:
    from parser import make_parser, make_preprocessor, parse_papers, qrels_doc_ids, select_papers

    qrel_ids = qrels_doc_ids(s.qrels_path)
    paths = select_papers(s.papers_dir, qrel_ids, s.max_papers, s.max_negatives)
    log.event("step_start", parser=s.parser, preprocessor=s.preprocessor, papers_dir=str(s.papers_dir),
              max_papers=s.max_papers, max_negatives=s.max_negatives, selected=[p.name for p in paths])
    documents = parse_papers(paths, make_parser(s.parser), make_preprocessor(s.preprocessor), qrel_ids, log)
    if not documents:
        raise RuntimeError(f"No papers parsed from {s.papers_dir}")
    write_jsonl(out_path, documents)
    sources = {}
    for d in documents:
        sources[d.metadata.section_source] = sources.get(d.metadata.section_source, 0) + 1
    log.event("step_complete", documents=len(documents), skipped=len(paths) - len(documents),
              with_qrels=sum(d.metadata.has_qrels for d in documents), pages=sum(d.metadata.page_count for d in documents),
              characters=sum(len(d.content) for d in documents), section_sources=sources, output=str(out_path))
    print(f"  Parsed {len(documents)}/{len(paths)} papers; sections from {sources} → {out_path}")


def chunk_step(s: IngestSettings, method: str, documents: list[Document], out_path: Path, cfg_path: Path,
               log: Logger) -> None:
    from chunking import build_config, chunk_documents, make_chunker
    from embedding import make_embedder

    config = build_config(method, s.chunk_size, s.overlap, parser=s.parser)
    log.event("step_start", config=config.model_dump(), documents=len(documents))
    boundary = make_embedder(s.semantic_model) if config.chunker == "semantic" else None
    chunks = chunk_documents(documents, make_chunker(config, boundary))
    write_jsonl(out_path, chunks)
    save_chunking_config(cfg_path, config)
    tokens = [c.metadata.token_count for c in chunks]
    stats = {
        "chunks": len(chunks),
        "min_tokens": min(tokens, default=0),
        "max_tokens": max(tokens, default=0),
        "avg_tokens": round(sum(tokens) / len(tokens), 1) if tokens else 0,
        "cross_page_chunks": sum(c.metadata.end_page_number > c.metadata.page_number for c in chunks),
    }
    log.event("step_complete", **stats, output=str(out_path), config_output=str(cfg_path))
    print(f"  {stats['chunks']} chunks (tokens min/avg/max: {stats['min_tokens']}/{stats['avg_tokens']}/"
          f"{stats['max_tokens']}) → {out_path}")


def embed_step(chunks: list[Chunk], model: str, out_path: Path, log: Logger) -> None:
    from embedding import embed_chunks, make_embedder, truncated_count

    embedder = make_embedder(model)
    log.event("step_start", chunks=len(chunks))
    vectors = embed_chunks(chunks, embedder, log)
    save_embeddings(out_path, chunks, vectors)
    truncated = truncated_count(embedder, [c.content for c in chunks])
    log.event("step_complete", chunks=len(chunks), dimensions=int(vectors.shape[1]), truncated=truncated,
              max_tokens=embedder.max_tokens, output=str(out_path))
    print(f"  {len(chunks)} chunks → {vectors.shape[1]} dims → {out_path}")
    if truncated:
        print(f"  WARNING: {truncated}/{len(chunks)} chunks exceed {model}'s {embedder.max_tokens}-token limit "
              f"and are truncated when embedded")


def index_step(index_name: str, chunks: list[Chunk], out_path: Path, log: Logger, vectors=None) -> None:
    from indexing import BM25Index, build_index, make_vector_index

    index = BM25Index() if index_name == "bm25" else make_vector_index(index_name)
    log.event("step_start", index=index.name, chunks=len(chunks))
    build_index(index, chunks, out_path, vectors)
    log.event("step_complete", index=index.name, entries=len(index), bytes=out_path.stat().st_size,
              output=str(out_path))
    print(f"  {index.name} index over {len(index)} chunks → {out_path}")


# ── Phases ────────────────────────────────────────────────────────────────────

def ingest(s: IngestSettings) -> list[TaskResult]:
    """Run every ingestion task; write and return the per-task summary."""
    it, dd = s.iteration, s.data_dir
    runner = Runner(s)
    documents: Optional[list[Document]] = None

    def get_documents() -> list[Document]:
        nonlocal documents
        if documents is None:
            documents = read_jsonl(parsed_path(it, dd), Document)
        return documents

    p_path = parsed_path(it, dd)
    parse_task = runner.run("parsing", {}, [p_path], lambda log: parse_step(s, p_path, log), [])

    for method in s.methods:
        key = method_key(method)
        c_path, cfg_path = chunks_path(it, key, dd), chunking_config_path(it, key, dd)
        chunk_task = runner.run("chunking", {"method": key}, [c_path, cfg_path],
                                lambda log: chunk_step(s, method, get_documents(), c_path, cfg_path, log),
                                [parse_task])
        chunks = functools.cache(lambda path=c_path: read_jsonl(path, Chunk))

        b_path = bm25_index_path(it, key, dd)
        runner.run("indexing", {"method": key, "index": "bm25"}, [b_path],
                   lambda log: index_step("bm25", chunks(), b_path, log), [chunk_task])

        for model in s.models:
            e_path = embeddings_path(it, key, model, dd)
            embed_task = runner.run("embedding", {"method": key, "model": model}, [e_path],
                                    lambda log: embed_step(chunks(), model, e_path, log), [chunk_task])
            f_path = faiss_index_path(it, key, model, dd)
            runner.run("indexing", {"method": key, "model": model, "index": s.vector_db.lower()}, [f_path],
                       lambda log: index_step(s.vector_db, chunks(), f_path, log,
                                              load_embeddings(e_path, chunks())),
                       [embed_task])

    save_json(pipeline_summary_path(it, dd), {
        "iteration": it,
        "tasks": [{"label": r.label, "step": r.step, **r.keys, "status": r.status,
                   "duration_seconds": r.duration_seconds, "outputs": r.outputs, "error": r.error}
                  for r in runner.results],
    })
    return runner.results


def query(args: argparse.Namespace) -> int:
    from generation import LLMGenerator
    from query_phase import QuerySettings, answer_question, find_configuration, run_query_phase

    s = QuerySettings(iteration=args.iteration, reranker=args.reranker, force=args.force,
                      reselect_queries=args.reselect_queries)
    generator = LLMGenerator(args.generator)
    if args.question is None:
        files = run_query_phase(s, generator)
        print(f"Query phase: {len(files)} answer files")
        return 0
    if not args.config:
        print("A single question needs --config, e.g. --config recursive-mpnet-hybrid")
        return 2
    entry = answer_question(args.question, find_configuration(s, args.config), s, generator)
    if entry.error:
        print(f"Error: {entry.error}")
        return 1
    print(entry.response.answer)
    for i, c in enumerate(entry.response.citations, 1):
        print(f"  [{i}] {c.source} p.{c.page_number} (score {c.relevance_score:.3f}): {c.text_snippet}")
    print(f"confidence {entry.response.confidence:.2f}")
    return 0


def evaluate(args: argparse.Namespace) -> int:
    from evaluate_phase import EvalSettings, best_result, compare_iterations, evaluate_iteration
    from evaluation import LLMAnswerJudge
    from query_phase import QuerySettings
    from saving import visualizations_dir

    if args.compare_to is not None and not (args.change and args.reason):
        print("--compare-to needs --change and --reason for the iteration log")
        return 2
    s = EvalSettings(query=QuerySettings(iteration=args.iteration, reranker=args.reranker), generator=args.generator,
                     measure_latency=not args.no_latency)
    results = evaluate_iteration(s, None if args.no_judge else LLMAnswerJudge(args.judge))
    best = best_result(results)
    print(f"Evaluated {len(results)} configurations; best (NDCG@{s.k}, then MRR): {best.experiment_id} "
          + ", ".join(f"{m} {best.metrics[m]:.3f}" for m in ("recall", "precision", "mrr", "ndcg")))
    if args.compare_to is not None:
        chart = compare_iterations(args.compare_to, args.iteration, args.change, args.reason, s.query.data_dir)
        print(f"Before/after → {chart}")
    print(f"Charts → {visualizations_dir(args.iteration, s.query.data_dir)}")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RAG pipeline: ingest, query, evaluate.")
    sub = parser.add_subparsers(dest="phase", required=True)

    ing = sub.add_parser("ingest", help="Parse, chunk, embed, and index the papers.")
    ing.add_argument("--force", action="store_true", help="Rerun every task even if its outputs exist")
    ing.add_argument("--only", choices=STEP_NAMES, help="Run just this step (all its tasks)")
    ing.add_argument("--iteration", type=int, default=ITERATION,
                     help=f"Iteration number used in output filenames (default: {ITERATION} from .env)")
    ing.add_argument("--max-papers", type=int, default=MAX_PAPERS,
                     help=f"Papers with qrels to ingest (default: {MAX_PAPERS} from MAX_PAPERS)")
    ing.add_argument("--max-negatives", type=int, default=MAX_NEGATIVES,
                     help=f"Hard-negative papers (no qrels) to add (default: {MAX_NEGATIVES} from MAX_NEGATIVES)")
    ing.add_argument("--methods", nargs="+", default=list(CHUNK_METHODS),
                     help=f"Chunking methods (default: {' '.join(CHUNK_METHODS)} from CHUNK_METHOD_1..3)")
    ing.add_argument("--models", nargs="+", default=list(EMBEDDING_MODELS),
                     help=f"Embedding models (default: {' '.join(EMBEDDING_MODELS)} from EMBEDDING_MODEL_1..2)")
    ing.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help=f"Tokens per chunk (default: {CHUNK_SIZE})")
    ing.add_argument("--overlap", type=int, default=OVERLAP, help=f"Tokens of overlap (default: {OVERLAP})")

    q = sub.add_parser("query", help="Answer the query set with every configuration, or one question.")
    q.add_argument("question", nargs="?", help="Answer just this question (needs --config)")
    q.add_argument("--config", help="Configuration for a single question, e.g. recursive-mpnet-hybrid")
    q.add_argument("--generator", default=GENERATION_MODEL,
                   help=f"Generator model (default: {GENERATION_MODEL} from GENERATION_MODEL)")
    q.add_argument("--reranker", default=RERANKER, help=f"none | cross-encoder | cohere (default: {RERANKER})")
    q.add_argument("--iteration", type=int, default=ITERATION,
                   help=f"Iteration number used in input/output filenames (default: {ITERATION} from .env)")
    q.add_argument("--force", action="store_true", help="Regenerate answer files that already exist")
    q.add_argument("--reselect-queries", action="store_true", help="Select a new query set")
    ev = sub.add_parser("evaluate", help="Score the iteration's answers: metrics, judge, latency, charts.")
    ev.add_argument("--iteration", type=int, default=ITERATION,
                    help=f"Iteration to evaluate (default: {ITERATION} from .env)")
    ev.add_argument("--generator", default=GENERATION_MODEL,
                    help=f"Whose answers to score (default: {GENERATION_MODEL} from GENERATION_MODEL)")
    ev.add_argument("--reranker", default=RERANKER, help=f"none | cross-encoder | cohere (default: {RERANKER})")
    ev.add_argument("--judge", default=JUDGE_MODEL, help=f"LLM judge (default: {JUDGE_MODEL} from JUDGE_MODEL)")
    ev.add_argument("--no-judge", action="store_true", help="Skip LLM judging (no radar chart)")
    ev.add_argument("--no-latency", action="store_true", help="Skip re-timing retrieval")
    ev.add_argument("--compare-to", type=int, help="Earlier iteration for the before/after chart and iteration log")
    ev.add_argument("--change", help="What changed since --compare-to (for the iteration log)")
    ev.add_argument("--reason", help="Why it was changed (for the iteration log)")

    args = parser.parse_args(argv)
    if args.phase == "ingest":
        results = ingest(IngestSettings(
            iteration=args.iteration, max_papers=args.max_papers, max_negatives=args.max_negatives,
            methods=args.methods, models=args.models, chunk_size=args.chunk_size, overlap=args.overlap,
            force=args.force, only=args.only,
        ))
        counts = {status: sum(r.status == status for r in results) for status in ("ran", "skipped", "failed", "blocked")}
        print(f"Ingestion: {counts} → {pipeline_summary_path(args.iteration)}")
        return 1 if counts["failed"] or counts["blocked"] else 0
    if args.phase == "query":
        return query(args)
    return evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
