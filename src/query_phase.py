"""
Query phase: answer the query set with every chunking × embedding × retrieval configuration.

    query set (selected once, then reused) → for each configuration:
      retrieve top-K per query (reranked when RERANKER is set) → generate an answer with citations
      → data/answers/{iteration}/{generator}/answer-{config}.json

The configurations are 3 chunking methods × (BM25 + 2 embedding models × {dense, hybrid}) = 15;
BM25 uses no embeddings. A failed LLM call is recorded in its entry's `error` and the
run continues. An answer file that already exists is skipped, except that its failed
entries are retried (only those); `force` regenerates the whole file.

Ad hoc questions go through answer_question with one named configuration.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from config import (
    ANSWERS_FILE, CHUNK_METHODS, DATA_DIR, EMBEDDING_MODELS, HYBRID_ALPHA, HYBRID_CANDIDATES, ITERATION,
    LLM_CONCURRENCY, QRELS_FILE, QUERIES_FILE, QUERY_ABSTRACTIVE, QUERY_COUNT, QUERY_SEED, QUERY_SOURCE_MIX,
    RERANK_CANDIDATES, RERANKER, RETRIEVAL_METHODS, TOP_K, Logger, method_key, resolve_project_path,
)
from generation import Generator
from models import (
    AnswerEntry, AnswerFile, Chunk, Document, QueryRecord, QuerySet, RetrievalConfig, RetrievalResult, RetrievedRef,
    RetrieverType,
)
from saving import (
    answer_path, bm25_index_path, chunks_path, faiss_index_path, load_model, parsed_path, query_set_path, read_jsonl,
    save_model,
)


@dataclass
class QuerySettings:
    iteration: int = ITERATION
    methods: list[str] = field(default_factory=lambda: list(CHUNK_METHODS))
    models: list[str] = field(default_factory=lambda: list(EMBEDDING_MODELS))
    retrieval_methods: list[str] = field(default_factory=lambda: list(RETRIEVAL_METHODS))
    top_k: int = TOP_K
    hybrid_alpha: float = HYBRID_ALPHA
    hybrid_candidates: int = HYBRID_CANDIDATES
    reranker: str = RERANKER
    rerank_candidates: int = RERANK_CANDIDATES
    query_count: int = QUERY_COUNT
    query_abstractive: int = QUERY_ABSTRACTIVE
    query_source_mix: str = QUERY_SOURCE_MIX
    query_seed: int = QUERY_SEED
    queries_path: Path = field(default_factory=lambda: resolve_project_path(QUERIES_FILE))
    qrels_path: Path = field(default_factory=lambda: resolve_project_path(QRELS_FILE))
    answers_path: Path = field(default_factory=lambda: resolve_project_path(ANSWERS_FILE))
    concurrency: int = LLM_CONCURRENCY
    force: bool = False                 # regenerate answer files that already exist
    reselect_queries: bool = False      # select a new query set even if one exists
    data_dir: Path = DATA_DIR
    log_path: Optional[Path] = None


def configurations(s: QuerySettings) -> list[RetrievalConfig]:
    """BM25 once per chunking method; dense and hybrid per chunking method × embedding model."""
    common = dict(top_k=s.top_k, reranker=s.reranker.lower())
    configs = []
    for method in s.methods:
        key = method_key(method)
        for retrieval in s.retrieval_methods:
            kind = RetrieverType(retrieval.lower())
            if kind is RetrieverType.BM25:
                configs.append(RetrievalConfig(chunk_method=key, retrieval_method=kind, **common))
                continue
            for model in s.models:
                configs.append(RetrievalConfig(chunk_method=key, embedding_model=model, retrieval_method=kind,
                                               hybrid_alpha=s.hybrid_alpha if kind is RetrieverType.HYBRID else None,
                                               **common))
    return configs


def find_configuration(s: QuerySettings, name: str) -> RetrievalConfig:
    configs = {c.name: c for c in configurations(s)}
    if name not in configs:
        raise ValueError(f"Unknown configuration {name!r}; expected one of {', '.join(configs)}")
    return configs[name]


def load_query_set(s: QuerySettings, log: Optional[Logger] = None) -> QuerySet:
    """The iteration's query set: read from disk, or selected and saved on first use (or with
    `reselect_queries`). Kept fixed otherwise, so reruns answer the same questions."""
    from query_set import build_query_set, parse_mix

    path = query_set_path(s.iteration, s.data_dir)
    if path.exists() and not s.reselect_queries:
        return load_model(path, QuerySet)
    doc_ids = {d.metadata.arxiv_id for d in read_jsonl(parsed_path(s.iteration, s.data_dir), Document)}
    query_set = build_query_set(s.queries_path, s.qrels_path, s.answers_path, doc_ids, s.query_count,
                                s.query_abstractive, parse_mix(s.query_source_mix), s.query_seed)
    save_model(path, query_set)
    if log:
        log.event("query_set_created", queries=len(query_set.queries), output=str(path),
                  query_ids=[q.query_id for q in query_set.queries])
    return query_set


class Resources:
    """Chunks, embedders, and retrievers, loaded once per run."""

    def __init__(self, s: QuerySettings) -> None:
        self.s = s
        self._chunks: dict[str, list[Chunk]] = {}

    def chunks(self, method: str) -> list[Chunk]:
        if method not in self._chunks:
            self._chunks[method] = read_jsonl(chunks_path(self.s.iteration, method, self.s.data_dir), Chunk)
        return self._chunks[method]

    def retriever(self, config: RetrievalConfig):
        import embedding
        from retrieval import make_reranker, make_retriever

        it, dd, key = self.s.iteration, self.s.data_dir, config.chunk_method
        model = config.embedding_model
        return make_retriever(
            config.retrieval_method, self.chunks(key), bm25_path=bm25_index_path(it, key, dd),
            faiss_path=faiss_index_path(it, key, model, dd) if model else None,
            embedder=embedding.make_embedder(model) if model else None, alpha=self.s.hybrid_alpha,
            candidates=self.s.hybrid_candidates, reranker=make_reranker(config.reranker),
            rerank_candidates=self.s.rerank_candidates,
        )


def _refs(results: list[RetrievalResult]) -> list[RetrievedRef]:
    return [RetrievedRef(rank=i, chunk_id=r.chunk.id, score=r.score, retriever_type=r.retriever_type,
                         source=r.chunk.metadata.source, page_number=r.chunk.metadata.page_number,
                         section_title=r.chunk.metadata.section_title) for i, r in enumerate(results, 1)]


def answer_entry(query: QueryRecord | str, results: list[RetrievalResult], retrieval_seconds: float,
                 generator: Generator) -> AnswerEntry:
    """Generate one answer; an LLM failure is recorded rather than raised."""
    query_id, text = (query.query_id, query.query) if isinstance(query, QueryRecord) else ("adhoc", query)
    entry = AnswerEntry(query_id=query_id, query=text, retrieved=_refs(results),
                        retrieval_seconds=round(retrieval_seconds, 3))
    start = time.monotonic()
    try:
        entry.response, usage = generator.generate_with_usage(text, results)
        entry.prompt_tokens, entry.completion_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
    except Exception as e:
        entry.error = f"{type(e).__name__}: {e}"
    entry.generation_seconds = round(time.monotonic() - start, 3)
    return entry


def answer_configuration(config: RetrievalConfig, queries: list[QueryRecord], resources: Resources,
                         generator: Generator, iteration: int, concurrency: int,
                         existing: Optional[AnswerFile] = None) -> AnswerFile:
    """Retrieve for every query (sequentially: local models), then generate in parallel.
    With `existing`, its successful entries are kept and only the rest are answered."""
    kept = {e.query_id: e for e in existing.entries if e.error is None} if existing else {}
    todo = [q for q in queries if q.query_id not in kept]
    retriever = resources.retriever(config)
    retrieved = []
    for q in todo:
        start = time.monotonic()
        retrieved.append((retriever.retrieve(q.query, config.top_k), time.monotonic() - start))
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        new = {e.query_id: e for e in pool.map(lambda args: answer_entry(args[0], *args[1], generator),
                                               zip(todo, retrieved))}
    return AnswerFile(iteration=iteration, config=config, generator=generator.name,
                      created_at=datetime.now(UTC).isoformat(),
                      entries=[kept.get(q.query_id) or new[q.query_id] for q in queries])


def run_query_phase(s: QuerySettings, generator: Generator) -> list[Path]:
    """Answer the query set with every configuration; returns the answer files (written or already there)."""
    log = Logger(s.iteration, "query", path=s.log_path, generator=generator.name)
    try:
        queries = load_query_set(s, log).queries
        resources = Resources(s)
        log.event("step_start", queries=len(queries), configurations=len(configurations(s)), reranker=s.reranker)
        paths = []
        for config in configurations(s):
            path = answer_path(s.iteration, generator.name, config, s.data_dir)
            paths.append(path)
            existing = load_model(path, AnswerFile) if path.exists() and not s.force else None
            if existing and all(e.error is None for e in existing.entries) \
                    and {e.query_id for e in existing.entries} == {q.query_id for q in queries}:
                print(f"[query / {generator.name} / {config.name}] skipped")
                continue
            start = time.monotonic()
            answers = answer_configuration(config, queries, resources, generator, s.iteration, s.concurrency,
                                           existing)
            save_model(path, answers)
            errors = sum(e.error is not None for e in answers.entries)
            log.event("configuration_complete", configuration=config.name, answers=len(answers.entries),
                      errors=errors, output=str(path), duration_seconds=round(time.monotonic() - start, 2))
            print(f"[query / {generator.name} / {config.name}] {len(answers.entries) - errors}/{len(answers.entries)} "
                  f"answered ({time.monotonic() - start:.1f}s) → {path}")
        log.event("step_complete", answer_files=len(paths))
        return paths
    finally:
        log.close()


def answer_question(question: str, config: RetrievalConfig, s: QuerySettings, generator: Generator) -> AnswerEntry:
    """One ad hoc question through one configuration."""
    retriever = Resources(s).retriever(config)
    start = time.monotonic()
    results = retriever.retrieve(question, config.top_k)
    return answer_entry(question, results, time.monotonic() - start, generator)
