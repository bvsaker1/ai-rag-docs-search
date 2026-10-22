import json
from collections import Counter
from pathlib import Path

import pytest

import embedding
import pipeline
from conftest import FakeEmbedder
from evaluation import AnswerJudge, ExperimentTracker, MetricsCalculator
from generation import Generator
from pipeline import IngestSettings, ingest
from retrieval import Retriever
from saving import chunks_path, embeddings_path, faiss_index_path, pipeline_summary_path

MODELS = ["fake-a", "fake-b"]
METHODS = ["FIXED", "RECURSIVE", "SEMANTIC"]


@pytest.fixture
def fake_models(monkeypatch):
    """Every embedding model is a FakeEmbedder; models named in `broken` fail to embed."""
    embedders: dict[str, FakeEmbedder] = {}
    broken: set[str] = set()

    def make(name: str, backend: str = "sentence-transformers"):
        if name in broken:
            raise RuntimeError(f"cannot load {name}")
        return embedders.setdefault(name, FakeEmbedder(name))

    monkeypatch.setattr(embedding, "make_embedder", make)
    return broken


@pytest.fixture
def settings(tmp_path, papers_dir, qrels_file, fake_models) -> IngestSettings:
    data = tmp_path / "data"
    data.mkdir()
    return IngestSettings(iteration=1, papers_dir=papers_dir, qrels_path=qrels_file, max_papers=50, max_negatives=1,
                          methods=METHODS, models=MODELS, chunk_size=40, overlap=8, semantic_model="fake-a",
                          data_dir=data, log_path=tmp_path / "log.jsonl")


def statuses(results) -> Counter:
    return Counter((r.step, r.status) for r in results)


def test_first_run_builds_every_artifact_and_summary(settings):
    results = ingest(settings)
    # 1 parse + per method: chunk, bm25, and per model: embed, faiss
    assert len(results) == 1 + 3 * (2 + 2 * 2)
    assert all(r.status == "ran" for r in results), [(r.label, r.error) for r in results if r.status != "ran"]
    for r in results:
        assert all(Path(p).exists() for p in r.outputs)
    summary = json.loads(pipeline_summary_path(1, settings.data_dir).read_text())
    assert len(summary["tasks"]) == len(results)
    events = [json.loads(line)["event"] for line in settings.log_path.read_text().splitlines()]
    assert events.count("step_complete") == len(results)


def test_parsing_takes_qrels_papers_then_negatives(settings):
    ingest(settings)
    log = [json.loads(line) for line in settings.log_path.read_text().splitlines()]
    start = next(e for e in log if e["step"] == "parsing" and e["event"] == "step_start")
    assert start["selected"] == ["2401.00001v1.pdf", "2401.00002v1.pdf", "2401.00003v1.pdf"]


def test_second_run_skips_everything(settings):
    ingest(settings)
    assert all(r.status == "skipped" for r in ingest(settings))


def test_only_reruns_a_step_and_its_dependents(settings):
    ingest(settings)
    settings.only = "chunking"
    counts = statuses(ingest(settings))
    assert counts[("parsing", "skipped")] == 1
    assert counts[("chunking", "ran")] == 3
    assert counts[("embedding", "skipped")] == 6 and counts[("indexing", "skipped")] == 9


def test_force_reruns_everything(settings):
    ingest(settings)
    settings.force = True
    assert all(r.status == "ran" for r in ingest(settings))


def test_a_missing_output_reruns_only_that_task_and_its_dependents(settings):
    ingest(settings)
    embeddings_path(1, "recursive", "fake-b", settings.data_dir).unlink()
    ran = [r.label for r in ingest(settings) if r.status == "ran"]
    assert ran == ["embedding / recursive / fake-b", "indexing / recursive / fake-b / faiss"]


def test_a_failed_task_blocks_its_dependents_only(settings, fake_models):
    fake_models.add("fake-b")
    results = {r.label: r for r in ingest(settings)}
    assert results["embedding / fixed / fake-b"].status == "failed"
    assert "cannot load fake-b" in results["embedding / fixed / fake-b"].error
    assert results["indexing / fixed / fake-b / faiss"].status == "blocked"
    assert results["indexing / fixed / fake-a / faiss"].status == "ran"
    assert not faiss_index_path(1, "fixed", "fake-b", settings.data_dir).exists()
    assert chunks_path(1, "fixed", settings.data_dir).exists()


def test_no_parsable_papers_fails_parsing_and_blocks_everything(settings, tmp_path):
    settings.papers_dir = tmp_path / "empty"
    settings.papers_dir.mkdir()
    results = ingest(settings)
    assert results[0].status == "failed" and all(r.status == "blocked" for r in results[1:])


def test_evaluate_compare_needs_change_and_reason(capsys):
    assert pipeline.main(["evaluate", "--iteration", "2", "--compare-to", "1"]) == 2
    assert "--change and --reason" in capsys.readouterr().out


def test_a_single_question_needs_a_configuration(capsys, monkeypatch):
    monkeypatch.setattr("generation.LLMGenerator.__init__", lambda self, model, client=None: None)
    assert pipeline.main(["query", "What is VDF-RLS?"]) == 2
    assert "--config" in capsys.readouterr().out


def test_query_and_evaluation_components_are_abstract():
    for abstract in (Retriever, Generator, MetricsCalculator, AnswerJudge, ExperimentTracker):
        with pytest.raises(TypeError):
            abstract()
