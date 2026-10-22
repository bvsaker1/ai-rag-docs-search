import json
from collections import Counter

import pytest

import embedding
from compare_generators import compare, provider
from models import Judgment
from conftest import FakeEmbedder, FakeLLMClient, cite_first_source, perfect_scores
from evaluation import LLMAnswerJudge
from generation import LLMGenerator
from models import AnswerFile, QuerySet
from pipeline import IngestSettings, ingest
from query_phase import (
    QuerySettings, answer_question, configurations, find_configuration, load_query_set, run_query_phase,
)
from saving import answer_path, generator_comparison_path, judgments_path, load_model, query_set_path, read_jsonl

PAPERS = ["2401.00001v1", "2401.00002v1", "2401.00003v1"]


@pytest.fixture
def fake_models(monkeypatch):
    embedders = {}
    monkeypatch.setattr(embedding, "make_embedder",
                        lambda name, backend="sentence-transformers": embedders.setdefault(name, FakeEmbedder(name)))


@pytest.fixture
def benchmark(tmp_path):
    """queries/qrels/answers.json with two queries per generated paper."""
    queries, qrels, answers = {}, {}, {}
    for p, paper in enumerate(PAPERS):
        for i, (qtype, topic) in enumerate([("abstractive", "estimate"), ("extractive", "data")]):
            qid = f"q{p}{i}"
            queries[qid] = {"query": f"How is the {topic} measured?", "type": qtype, "source": "text"}
            qrels[qid] = {"doc_id": paper, "section_id": 1}
            answers[qid] = f"Reference answer {qid}."
    paths = {}
    for name, data in [("queries", queries), ("qrels", qrels), ("answers", answers)]:
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(data))
    return paths


@pytest.fixture
def settings(tmp_path, papers_dir, benchmark, fake_models) -> QuerySettings:
    data = tmp_path / "data"
    data.mkdir()
    ingest(IngestSettings(iteration=1, papers_dir=papers_dir, qrels_path=benchmark["qrels"], max_papers=3,
                          models=["fake-a", "fake-b"], chunk_size=40, overlap=8, semantic_model="fake-a",
                          data_dir=data, log_path=tmp_path / "log.jsonl"))
    return QuerySettings(iteration=1, models=["fake-a", "fake-b"], top_k=3, query_count=3, query_abstractive=2,
                         query_source_mix="text:1.0", queries_path=benchmark["queries"],
                         qrels_path=benchmark["qrels"], answers_path=benchmark["answers"], concurrency=2,
                         data_dir=data, log_path=tmp_path / "log.jsonl")


def generator(model: str = "vendor/gen-a", **kw) -> LLMGenerator:
    return LLMGenerator(model, client=FakeLLMClient(cite_first_source, **kw), sleep=lambda s: None)


def test_fifteen_configurations_with_bm25_once_per_chunking_method():
    configs = configurations(QuerySettings(models=["all-MiniLM-L6-v2", "all-mpnet-base-v2"]))
    names = [c.name for c in configs]
    assert len(configs) == 15
    assert Counter(c.retrieval_method.value for c in configs) == {"bm25": 3, "dense": 6, "hybrid": 6}
    assert {"fixed-bm25", "recursive-mpnet-hybrid", "semantic-minilm-dense"} <= set(names)
    assert all(c.hybrid_alpha is None for c in configs if c.retrieval_method.value != "hybrid")
    assert configurations(QuerySettings(reranker="cohere"))[0].name == "fixed-bm25-rerank-cohere"
    with pytest.raises(ValueError, match="Unknown configuration"):
        find_configuration(QuerySettings(), "fixed-mpnet-bm25")


def test_query_set_is_selected_once_from_ingested_papers_and_reused(settings):
    first = load_query_set(settings)
    assert len(first.queries) == 3 and len({q.doc_id for q in first.queries}) == 3
    assert Counter(q.type for q in first.queries) == {"abstractive": 2, "extractive": 1}
    assert all(q.reference_answer.startswith("Reference answer") for q in first.queries)
    saved = load_model(query_set_path(1, settings.data_dir), QuerySet)
    assert saved == first
    settings.query_seed = 99
    assert load_query_set(settings) == first                 # kept fixed without reselect_queries


def test_query_phase_writes_one_answer_file_per_configuration(settings):
    paths = run_query_phase(settings, generator())
    assert len(paths) == 15 and all(p.exists() for p in paths)
    assert paths[0].parent.name == "gen-a"
    assert answer_path(1, "vendor/gen-a", find_configuration(settings, "recursive-fake-a-hybrid"),
                       settings.data_dir).name == "answer-recursive-fake-a-hybrid.json"
    answers = load_model(paths[0], AnswerFile)
    assert answers.config.name == "fixed-bm25" and answers.generator == "vendor/gen-a"
    for entry in answers.entries:
        assert entry.error is None and len(entry.retrieved) == 3
        assert [r.rank for r in entry.retrieved] == [1, 2, 3]
        assert entry.response.citations[0].chunk_id == entry.retrieved[0].chunk_id
        assert entry.prompt_tokens == 100


def test_query_phase_skips_existing_files_unless_forced(settings):
    run_query_phase(settings, generator())
    client_calls = generator()
    run_query_phase(settings, client_calls)
    assert client_calls.client.calls == []
    settings.force = True
    run_query_phase(settings, client_calls)
    assert len(client_calls.client.calls) == 15 * 3


def test_llm_failures_are_recorded_per_entry(settings):
    failing = LLMGenerator("vendor/gen-x", client=FakeLLMClient(cite_first_source, fail_times=10 ** 6),
                           max_attempts=1, sleep=lambda s: None)
    paths = run_query_phase(settings, failing)
    entry = load_model(paths[0], AnswerFile).entries[0]
    assert entry.response is None and "network down" in entry.error and entry.retrieved


def test_answer_question_runs_one_configuration(settings):
    entry = answer_question("How is the estimate measured?", find_configuration(settings, "semantic-fake-b-dense"),
                            settings, generator())
    assert entry.query_id == "adhoc" and entry.response.citations and len(entry.retrieved) == 3


def test_compare_generators_judges_every_answer_and_ranks_generators(settings):
    def make(model):
        return generator(model) if model != "vendor/broken" else generator(model, fail_times=10 ** 6)

    judge = LLMAnswerJudge("judge-co/judge", client=FakeLLMClient(perfect_scores), sleep=lambda s: None)
    summary = compare(["vendor/gen-a", "other/gen-b", "vendor/broken"], settings, make, judge)
    assert set(summary) == {"vendor/gen-a", "other/gen-b"}             # broken failed its smoke test
    row = summary["vendor/gen-a"]
    assert row["answers"] == row["judged"] == 45 and row["overall"] == 4.5 and row["citation_quality"] == 5
    assert row["overall_abstractive"] == 4.5 and row["answer_errors"] == 0
    saved = json.loads(generator_comparison_path(1, settings.data_dir).read_text())
    assert saved["judge"] == "judge-co/judge" and len(saved["configurations"]) == 15
    assert len(read_jsonl(judgments_path(1, settings.data_dir), Judgment)) == 90

    rejudge = LLMAnswerJudge("judge-co/judge", client=FakeLLMClient(perfect_scores))
    compare(["vendor/gen-a"], settings, make, rejudge)
    assert rejudge.client.calls == []                                  # cached judgments reused


def test_provider():
    assert provider("anthropic/claude-haiku-5.5") == "anthropic" and provider("local") == "local"


def test_existing_files_retry_only_their_failed_entries(settings):
    failing = LLMGenerator("vendor/gen-a", client=FakeLLMClient(cite_first_source, fail_times=1), max_attempts=1,
                           sleep=lambda s: None)
    settings.concurrency = 1
    paths = run_query_phase(settings, failing)
    first = load_model(paths[0], AnswerFile)
    failed = [e.query_id for e in first.entries if e.error]
    assert len(failed) == 1

    retry = generator()
    run_query_phase(settings, retry)
    assert len(retry.client.calls) == 1                                 # only the failed entry
    again = load_model(paths[0], AnswerFile)
    assert all(e.error is None for e in again.entries)
    assert [e.query_id for e in again.entries] == [e.query_id for e in first.entries]
    unchanged = [e for e in first.entries if e.query_id not in failed]
    assert all(e in again.entries for e in unchanged)


def test_regenerated_answers_are_judged_again(settings):
    judge = LLMAnswerJudge("judge-co/judge", client=FakeLLMClient(perfect_scores))
    compare(["vendor/gen-a"], settings, generator, judge)
    first_calls = len(judge.client.calls)
    settings.force = True
    compare(["vendor/gen-a"], settings, generator, judge)
    # same fake answers → same hashes → cached; force alone doesn't re-judge identical answers
    assert len(judge.client.calls) == first_calls

    def different(model, messages, response_model):
        raw = cite_first_source(model, messages, response_model)
        return {**raw, "answer": "A different answer [1]."}

    compare(["vendor/gen-a"], settings, lambda m: LLMGenerator(m, client=FakeLLMClient(different)), judge)
    assert len(judge.client.calls) == 2 * first_calls
