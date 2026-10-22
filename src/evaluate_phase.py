"""
Evaluation phase: score every configuration's saved answers.

    query set + answer files (query phase) → ground truth per chunking method (corpus sections)
      → retrieval metrics at K from each answer's saved top-K
      → warm retrieval latency (re-timed after one warm-up query; no LLM) + saved generation latency
      → LLM-judge scores (cached, so already-judged answers cost nothing)
      → data/metrics_{iteration}.jsonl, data/best_config_{iteration}.json, logs/{LOG_BASE}_{iteration}.md
      → charts in data/visualizations/{iteration}/

The best configuration has the highest NDCG@K, then MRR. With `compare_to`, the best
configurations of both iterations are charted before/after and the change is appended to
logs/iteration_log.md and logs/iteration_log.jsonl (change, reason, metrics before/after, delta).
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
import time
from typing import Optional

from config import CORPUS_DIR, LOGS_DIR, METRIC_K, RELEVANCE_MIN_OVERLAP, RELEVANCE_NGRAM, Logger, resolve_project_path
from evaluation import AnswerJudge, IRMetricsCalculator, JsonlExperimentTracker, judge_answers, relevant_chunk_ids
from models import AnswerFile, EvaluationResult, Judgment, QueryMetrics, QueryRecord, RetrievalConfig
from query_phase import QuerySettings, Resources, configurations, load_query_set
from saving import (
    answer_path, best_config_path, chunking_config_path, iteration_log_path, judgments_path, load_model,
    markdown_log_path, metrics_path, read_jsonl, save_json, visualizations_dir,
)

CRITERIA = ["relevance", "accuracy", "completeness", "citation_quality"]
SECTION_METRICS = ["recall", "precision", "mrr", "ndcg", "hit", "max_precision"]
PAPER_METRICS = ["paper_precision", "paper_mrr", "paper_ndcg", "paper_hit"]
HEADLINE = ["recall", "precision", "mrr", "ndcg"]


@dataclass
class EvalSettings:
    query: QuerySettings = field(default_factory=QuerySettings)
    generator: str = ""
    k: int = METRIC_K
    corpus_dir: Path = field(default_factory=lambda: resolve_project_path(CORPUS_DIR))
    min_overlap: float = RELEVANCE_MIN_OVERLAP
    ngram: int = RELEVANCE_NGRAM
    measure_latency: bool = True
    logs_dir: Path = LOGS_DIR


def best_result(results: list[EvaluationResult]) -> EvaluationResult:
    return max(results, key=lambda r: (r.metrics["ndcg"], r.metrics["mrr"]))


def warm_latency_ms(resources: Resources, config: RetrievalConfig, queries: list[QueryRecord]) -> dict[str, float]:
    """query_id → retrieval milliseconds, timed after one warm-up query (models and indexes loaded).
    The query-vector cache is cleared before each query, so timing includes embedding the query."""
    import retrieval

    retriever = resources.retriever(config)
    retriever.retrieve(queries[0].query, config.top_k)
    times = {}
    for q in queries:
        retrieval.clear_query_vectors()
        start = time.perf_counter()
        retriever.retrieve(q.query, config.top_k)
        times[q.query_id] = (time.perf_counter() - start) * 1000
    return times


def score_answers(answers: AnswerFile, queries: dict[str, QueryRecord], relevant: dict[str, set[str]],
                  paper_chunks: dict[str, set[str]], k: int, latency_ms: dict[str, float],
                  judgments: dict[tuple[str, str], Judgment]) -> list[QueryMetrics]:
    """Per-query metrics from an answer file's saved top-K. Paper level counts every chunk of the
    qrels paper as relevant, so its ideal top K is K chunks of that paper."""
    calculator = IRMetricsCalculator()
    rows = []
    for entry in answers.entries:
        q = queries[entry.query_id]
        retrieved = [str(r.chunk_id) for r in entry.retrieved]
        section = calculator.score(retrieved, relevant[q.query_id], k)
        paper = calculator.score(retrieved, paper_chunks[q.doc_id], k)
        judgment = judgments.get((answers.config.name, q.query_id))
        rows.append(QueryMetrics(
            query_id=q.query_id, type=q.type, source=q.source, relevant_chunks=len(relevant[q.query_id]),
            **section, max_precision=min(k, len(relevant[q.query_id])) / k,
            paper_precision=paper["precision"], paper_mrr=paper["mrr"], paper_ndcg=paper["ndcg"],
            paper_hit=paper["hit"], retrieval_ms=latency_ms.get(q.query_id),
            generation_seconds=entry.generation_seconds if entry.response else None,
            judge=judgment.scores if judgment and judgment.scores else None,
        ))
    return rows


def summarize(rows: list[QueryMetrics]) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """(metric means, judge means, latency summary) over a configuration's queries."""
    metrics = {m: mean(getattr(r, m) for r in rows) for m in SECTION_METRICS + PAPER_METRICS}
    for qtype in ("abstractive", "extractive"):
        typed = [r for r in rows if r.type == qtype]
        if typed:
            metrics[f"ndcg_{qtype}"] = mean(r.ndcg for r in typed)
    judged = [r.judge for r in rows if r.judge]
    judge = {c: mean(getattr(j, c) for j in judged) for c in CRITERIA} if judged else {}
    if judge:
        judge["overall"] = mean(j.overall for j in judged)
        judge["judged"] = len(judged)
    latency = {}
    retrieval = [r.retrieval_ms for r in rows if r.retrieval_ms is not None]
    if retrieval:
        latency.update(retrieval_ms_median=median(retrieval), retrieval_ms_max=max(retrieval))
    total = [r.retrieval_ms / 1000 + r.generation_seconds for r in rows
             if r.retrieval_ms is not None and r.generation_seconds is not None]
    if total:
        latency.update(end_to_end_s_median=median(total), end_to_end_s_mean=mean(total), end_to_end_s_max=max(total))
    return metrics, judge, latency


def evaluate_iteration(s: EvalSettings, judge: Optional[AnswerJudge] = None) -> list[EvaluationResult]:
    """Score every configuration's answers for s.generator; write metrics, best config, report, charts."""
    qs = s.query
    log = Logger(qs.iteration, "evaluation", path=qs.log_path, generator=s.generator)
    try:
        queries = {q.query_id: q for q in load_query_set(qs, log).queries}
        configs = configurations(qs)
        files = {c.name: answer_path(qs.iteration, s.generator, c, qs.data_dir) for c in configs}
        missing = [name for name, p in files.items() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"No answers for {missing} — run `pipeline.py query --generator {s.generator}`")
        log.event("step_start", configurations=len(configs), queries=len(queries), k=s.k, min_overlap=s.min_overlap,
                  judge=judge.name if judge else None)

        resources = Resources(qs)
        relevant: dict[str, dict[str, set[str]]] = {}
        paper_chunks: dict[str, dict[str, set[str]]] = {}
        for method in {c.chunk_method for c in configs}:
            chunks = resources.chunks(method)
            relevant[method] = {qid: relevant_chunk_ids(q, chunks, s.corpus_dir, s.min_overlap, s.ngram)
                                for qid, q in queries.items()}
            by_paper = defaultdict(set)
            for c in chunks:
                by_paper[c.metadata.arxiv_id].add(str(c.id))
            paper_chunks[method] = by_paper
            log.event("ground_truth", chunk_method=method,
                      relevant_per_query={qid: len(ids) for qid, ids in relevant[method].items()})

        judgments: dict[tuple[str, str], Judgment] = {}
        if judge:
            for j in judge_answers(list(files.values()), queries, judge, judgments_path(qs.iteration, qs.data_dir),
                                   qs.concurrency):
                judgments[(j.configuration, j.query_id)] = j

        tracker = JsonlExperimentTracker(metrics_path(qs.iteration, qs.data_dir))
        results = []
        for config in configs:
            answers = load_model(files[config.name], AnswerFile)
            latency = warm_latency_ms(resources, config, list(queries.values())) if s.measure_latency else {}
            rows = score_answers(answers, queries, relevant[config.chunk_method], paper_chunks[config.chunk_method],
                                 s.k, latency, judgments)
            metrics, judge_scores, latency_summary = summarize(rows)
            chunking_file = chunking_config_path(qs.iteration, config.chunk_method, qs.data_dir)
            result = EvaluationResult(
                experiment_id=config.name, iteration=qs.iteration, config=config,
                chunking=json.loads(chunking_file.read_text()) if chunking_file.exists() else {},
                generator=s.generator, judge_model=judge.name if judge else None, k=s.k, metrics=metrics,
                judge_scores=judge_scores, latency=latency_summary, per_query=rows)
            tracker.record(result)
            results.append(result)
            log.event("configuration_scored", configuration=config.name, metrics=metrics, judge_scores=judge_scores,
                      latency=latency_summary)

        best = best_result(results)
        save_json(best_config_path(qs.iteration, qs.data_dir), {
            "iteration": qs.iteration, "experiment_id": best.experiment_id, "ranked_by": [f"ndcg@{s.k}", "mrr"],
            "config": best.config.model_dump(mode="json"), "chunking": best.chunking, "generator": best.generator,
            "metrics": best.metrics, "judge_scores": best.judge_scores, "latency": best.latency,
            "env": env_for(best),
        })
        write_report(results, best, s, judge.name if judge else None)
        charts = write_charts(results, best, visualizations_dir(qs.iteration, qs.data_dir))
        log.event("step_complete", best=best.experiment_id, metrics_output=str(metrics_path(qs.iteration, qs.data_dir)),
                  charts=[str(p) for p in charts])
        return results
    finally:
        log.close()


def env_for(result: EvaluationResult) -> dict[str, str]:
    """The .env settings that select this configuration."""
    c, chunking = result.config, result.chunking
    env = {"CHUNK_SIZE": str(chunking.get("chunk_size", "")), "OVERLAP": str(chunking.get("overlap", "")),
           "CHUNK_METHOD": c.chunk_method.upper(), "RETRIEVAL": c.retrieval_method.value, "TOP_K": str(c.top_k),
           "RERANKER": c.reranker, "GENERATION_MODEL": result.generator}
    if c.embedding_model:
        env["EMBEDDING_MODEL"] = c.embedding_model
    if c.hybrid_alpha is not None:
        env["HYBRID_ALPHA"] = str(c.hybrid_alpha)
    return env


# ── Outputs ───────────────────────────────────────────────────────────────────

def write_charts(results: list[EvaluationResult], best: EvaluationResult, out_dir: Path) -> list[Path]:
    import visualize

    charts = [visualize.retrieval_heatmap(results, out_dir / "retrieval_heatmap.png"),
              visualize.dimension_impact(results, out_dir / "dimension_impact.png")]
    if best.judge_scores:
        charts.append(visualize.generation_radar(best, out_dir / "generation_radar.png"))
    if any(r.latency for r in results):
        charts.append(visualize.query_latency(results, out_dir / "query_latency.png"))
    return charts


def _fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def write_report(results: list[EvaluationResult], best: EvaluationResult, s: EvalSettings,
                 judge_name: Optional[str]) -> Path:
    """Append this run's section to logs/{LOG_BASE}_{iteration}.md."""
    k = s.k
    lines = [f"# Evaluation run {datetime.now(UTC).isoformat(timespec='seconds')} "
             f"(iteration {s.query.iteration}, {len(results)} configurations, generator {s.generator})", "",
             f"- **Ground truth:** chunks overlapping the qrels section (word {s.ngram}-grams, either direction, "
             f"≥ {s.min_overlap:.0%})",
             f"- **Best configuration** (NDCG@{k}, then MRR): **{best.experiment_id}**",
             f"- **Judge:** {judge_name or 'not run'}", "",
             f"| Configuration | Recall@{k} | Precision@{k} (max) | MRR | NDCG@{k} | Hit@{k} | Paper P@{k} | "
             "Judge overall | Retrieval ms | End-to-end s |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: (-r.metrics["ndcg"], -r.metrics["mrr"])):
        m = r.metrics
        lines.append(f"| {r.experiment_id} | {m['recall']:.3f} | {m['precision']:.3f} ({m['max_precision']:.2f}) | "
                     f"{m['mrr']:.3f} | {m['ndcg']:.3f} | {m['hit']:.3f} | {m['paper_precision']:.3f} | "
                     f"{_fmt(r.judge_scores.get('overall', '—'))} | {_fmt(r.latency.get('retrieval_ms_median', '—'))} | "
                     f"{_fmt(r.latency.get('end_to_end_s_median', '—'))} |")
    if best.judge_scores:
        lines += ["", f"**Best configuration judge scores:** " + ", ".join(
            f"{c} {best.judge_scores[c]:.2f}" for c in CRITERIA) + f", overall {best.judge_scores['overall']:.2f}"]
    path = markdown_log_path(s.query.iteration, s.logs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")
    return path


def compare_iterations(before_iteration: int, after_iteration: int, change: str, reason: str,
                       data_dir: Path, logs_dir: Path = LOGS_DIR) -> Path:
    """Before/after chart of the two iterations' best configurations, and an iteration-log entry."""
    import visualize

    def best_of(iteration: int) -> EvaluationResult:
        path = metrics_path(iteration, data_dir)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — evaluate iteration {iteration} first")
        return best_result(read_jsonl(path, EvaluationResult))

    before, after = best_of(before_iteration), best_of(after_iteration)
    chart = visualize.before_after(before, after, visualizations_dir(after_iteration, data_dir) / "before_after.png",
                                   change)
    metrics = HEADLINE + ["hit"]
    entry = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"), "before_iteration": before_iteration,
        "after_iteration": after_iteration, "change": change, "reason": reason,
        "best_before": before.experiment_id, "best_after": after.experiment_id,
        "metric_before": {m: round(before.metrics[m], 4) for m in metrics},
        "metric_after": {m: round(after.metrics[m], 4) for m in metrics},
        "delta": {m: round(after.metrics[m] - before.metrics[m], 4) for m in metrics},
    }
    logs_dir.mkdir(parents=True, exist_ok=True)
    with open(iteration_log_path(logs_dir, ".jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    k = after.k
    names = {"recall": f"Recall@{k}", "precision": f"Precision@{k}", "mrr": "MRR", "ndcg": f"NDCG@{k}", "hit": f"Hit@{k}"}
    md = [f"## Iteration {before_iteration} → {after_iteration} ({entry['timestamp']})", "",
          f"- **Change:** {change}", f"- **Reason:** {reason}",
          f"- **Best configuration:** {before.experiment_id} → {after.experiment_id}", "",
          "| Metric | Before | After | Delta |", "|---|---|---|---|"]
    md += [f"| {names[m]} | {entry['metric_before'][m]:.3f} | {entry['metric_after'][m]:.3f} | "
           f"{entry['delta'][m]:+.3f} |" for m in metrics]
    with open(iteration_log_path(logs_dir), "a", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n\n")
    return chart
