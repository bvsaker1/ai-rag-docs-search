import json
import math

import pytest

from conftest import FakeLLMClient, perfect_scores
from evaluate_phase import EvalSettings, best_result, compare_iterations, evaluate_iteration, summarize
from evaluation import (
    IRMetricsCalculator, JsonlExperimentTracker, LLMAnswerJudge, MetricsCalculator, corpus_sections, hit_at_k,
    ndcg_at_k, ngrams, overlap, precision_at_k, recall_at_k, reciprocal_rank, relevant_chunk_ids,
)
from models import EvaluationResult, QueryMetrics, QueryRecord, RetrievalConfig
from query_phase import run_query_phase
from saving import best_config_path, iteration_log_path, markdown_log_path, metrics_path, read_jsonl, visualizations_dir
from test_models import make_chunk
from test_query_phase import PAPERS, benchmark, fake_models, generator, settings  # noqa: F401 (fixtures)

# ── Metrics (worked by hand) ──────────────────────────────────────────────────

RETRIEVED = ["a", "x", "b", "y", "z"]
RELEVANT = {"a", "b", "c"}


def test_metrics_on_a_known_ranking():
    assert recall_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(2 / 3)
    assert precision_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(2 / 5)
    assert reciprocal_rank(RETRIEVED, RELEVANT, 5) == 1.0
    assert reciprocal_rank(["x", "y", "b"], RELEVANT, 5) == pytest.approx(1 / 3)
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    ideal = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(dcg / ideal)
    assert hit_at_k(RETRIEVED, RELEVANT, 5) == 1.0


def test_metrics_respect_k_and_edge_cases():
    assert recall_at_k(RETRIEVED, RELEVANT, 2) == pytest.approx(1 / 3)
    assert precision_at_k(["x", "y"], RELEVANT, 5) == 0.0
    assert reciprocal_rank(["x", "y", "z", "q", "r", "a"], RELEVANT, 5) == 0.0      # first hit below K
    assert ndcg_at_k(["a"], {"a"}, 5) == 1.0                                        # ideal = min(K, relevant)
    assert recall_at_k(RETRIEVED, set(), 5) == 0.0 and ndcg_at_k(RETRIEVED, set(), 5) == 0.0
    assert hit_at_k(["x"], RELEVANT, 5) == 0.0


def test_metrics_calculator_is_abstract_and_ir_calculator_reports_all_metrics():
    with pytest.raises(TypeError):
        MetricsCalculator()
    assert set(IRMetricsCalculator().score(RETRIEVED, RELEVANT, 5)) == {"recall", "precision", "mrr", "ndcg", "hit"}


# ── Ground truth ──────────────────────────────────────────────────────────────

def test_overlap_counts_either_direction():
    section = ngrams("the estimate improves with more data over time")
    small_chunk = ngrams("the estimate improves with more data")
    big_chunk = ngrams("intro text here the estimate improves with more data over time and then many other "
                       "unrelated words follow in this long chunk about something else entirely")
    assert overlap(small_chunk, section) == 1.0                 # all of the chunk is in the section
    assert overlap(big_chunk, section) == 1.0                   # all of the section is in the chunk
    assert overlap(ngrams("nothing shared at all here"), section) == 0.0
    assert overlap(set(), section) == 0.0


def test_relevant_chunks_come_from_the_qrels_paper_and_section(tmp_path):
    (tmp_path / "p1.json").write_text(json.dumps({"sections": str([
        {"section_id": 0, "text": "# Abstract\n\nWe study grid impedance estimation."},
        {"section_id": 1, "text": "## Methods\n\nThe recursive least squares estimator uses a forgetting factor."}])}))
    corpus_sections.cache_clear()
    query = QueryRecord(query_id="q", query="?", type="extractive", source="text", doc_id="p1", section_id=1,
                        reference_answer="")
    text = "The recursive least squares estimator uses a forgetting factor."
    chunks = [make_chunk(text, end_char=len(text), arxiv_id="p1"),
              make_chunk("We study grid impedance estimation.", end_char=35, arxiv_id="p1"),
              make_chunk(text, end_char=len(text), arxiv_id="other-paper")]
    assert relevant_chunk_ids(query, chunks, tmp_path) == {str(chunks[0].id)}


# ── Tracking & summaries ──────────────────────────────────────────────────────

def result(name: str, ndcg: float, mrr: float, iteration: int = 1) -> EvaluationResult:
    return EvaluationResult(
        experiment_id=name, iteration=iteration, config=RetrievalConfig(chunk_method="fixed", retrieval_method="bm25",
                                                                        top_k=5),
        chunking={}, generator="g", k=5, per_query=[],
        metrics={"recall": 0.5, "precision": 0.2, "mrr": mrr, "ndcg": ndcg, "hit": 1.0})


def test_best_result_ranks_by_ndcg_then_mrr():
    assert best_result([result("a", 0.5, 0.9), result("b", 0.7, 0.1), result("c", 0.7, 0.4)]).experiment_id == "c"


def test_tracker_replaces_a_configuration_recorded_again(tmp_path):
    tracker = JsonlExperimentTracker(tmp_path / "metrics.jsonl")
    tracker.record(result("a", 0.1, 0.1))
    tracker.record(result("b", 0.2, 0.2))
    tracker.record(result("a", 0.9, 0.9))
    reloaded = JsonlExperimentTracker(tmp_path / "metrics.jsonl").results()
    assert [(r.experiment_id, r.metrics["ndcg"]) for r in reloaded] == [("a", 0.9), ("b", 0.2)]


def test_summarize_reports_means_by_type_and_latency():
    rows = [QueryMetrics(query_id=str(i), type=t, source="text", relevant_chunks=1, recall=r, precision=r / 5, mrr=r,
                         ndcg=r, hit=r, max_precision=0.2, paper_precision=1, paper_mrr=1, paper_ndcg=1, paper_hit=1,
                         retrieval_ms=10.0, generation_seconds=2.0)
            for i, (t, r) in enumerate([("abstractive", 1.0), ("abstractive", 0.0), ("extractive", 1.0)])]
    metrics, judge, latency = summarize(rows)
    assert metrics["ndcg"] == pytest.approx(2 / 3) and metrics["ndcg_abstractive"] == 0.5
    assert metrics["ndcg_extractive"] == 1.0 and judge == {}
    assert latency["retrieval_ms_median"] == 10.0 and latency["end_to_end_s_median"] == pytest.approx(2.01)


# ── End to end ────────────────────────────────────────────────────────────────

@pytest.fixture
def corpus(tmp_path):
    """Corpus sections matching the generated papers: section 1 is the body text about each paper's topic."""
    d = tmp_path / "corpus"
    d.mkdir()
    for paper, topic in zip(PAPERS, ["impedance", "enrollment", "particles"]):
        body = (f"The {topic} estimate improves with more data. We measure {topic} on a test grid. "
                "Results are reported for every case. Each case uses the same settings.")
        (d / f"{paper}.json").write_text(json.dumps({"sections": [
            {"section_id": 0, "text": "A Study of Grid Estimation"}, {"section_id": 1, "text": body}]}))
    corpus_sections.cache_clear()
    return d


def test_evaluate_iteration_writes_metrics_best_config_report_and_charts(settings, corpus, tmp_path):
    run_query_phase(settings, generator())
    judge = LLMAnswerJudge("judge-co/judge", client=FakeLLMClient(perfect_scores))
    s = EvalSettings(query=settings, generator="vendor/gen-a", k=3, corpus_dir=corpus, logs_dir=tmp_path / "logs")
    results = evaluate_iteration(s, judge)

    assert len(results) == 15 and all(len(r.per_query) == 3 for r in results)
    assert read_jsonl(metrics_path(1, settings.data_dir), EvaluationResult) == results
    for r in results:
        assert all(q.relevant_chunks >= 1 for q in r.per_query)                     # every query has ground truth
        assert 0 <= r.metrics["precision"] <= r.metrics["max_precision"] <= 1
        assert r.judge_scores["overall"] == 4.5 and r.latency["retrieval_ms_median"] > 0
        assert r.chunking["chunk_size"] == 40
    best = json.loads(best_config_path(1, settings.data_dir).read_text())
    assert best["experiment_id"] == best_result(results).experiment_id and "CHUNK_SIZE" in best["env"]
    assert "| Configuration | Recall@3" in markdown_log_path(1, s.logs_dir).read_text()
    charts = {p.name for p in visualizations_dir(1, settings.data_dir).glob("*.png")}
    assert charts == {"retrieval_heatmap.png", "dimension_impact.png", "generation_radar.png", "query_latency.png"}
    assert len(judge.client.calls) == 45

    again = LLMAnswerJudge("judge-co/judge", client=FakeLLMClient(perfect_scores))
    evaluate_iteration(s, again)
    assert again.client.calls == []                                                 # cached judgments


def test_evaluate_needs_answers_first(settings, corpus):
    with pytest.raises(FileNotFoundError, match="pipeline.py query"):
        evaluate_iteration(EvalSettings(query=settings, generator="vendor/none", corpus_dir=corpus))


def test_compare_iterations_charts_and_logs_the_change(tmp_path):
    from evaluation import JsonlExperimentTracker
    for iteration, ndcg in [(1, 0.5), (2, 0.7)]:
        tracker = JsonlExperimentTracker(metrics_path(iteration, tmp_path))
        tracker.record(result(f"best-{iteration}", ndcg, 0.6, iteration))
        tracker.record(result(f"worse-{iteration}", 0.1, 0.1, iteration))
    chart = compare_iterations(1, 2, "CHUNK_SIZE 512 → 256", "Chunks were truncated", tmp_path, tmp_path / "logs")
    assert chart.exists() and chart.name == "before_after.png"
    entry = json.loads(iteration_log_path(tmp_path / "logs", ".jsonl").read_text().splitlines()[-1])
    assert entry["best_before"] == "best-1" and entry["best_after"] == "best-2"
    assert entry["delta"]["ndcg"] == pytest.approx(0.2) and entry["reason"] == "Chunks were truncated"
    md = iteration_log_path(tmp_path / "logs").read_text()
    assert "**Change:** CHUNK_SIZE 512 → 256" in md and "| NDCG@5 | 0.500 | 0.700 | +0.200 |" in md
