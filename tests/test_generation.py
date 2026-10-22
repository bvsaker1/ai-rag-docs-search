import pytest

from conftest import FakeLLMClient, cite_first_source, perfect_scores
from evaluation import LLMAnswerJudge
from generation import (
    NO_SOURCES_ANSWER, CitationExtractor, GeneratedAnswer, Generator, LLMGenerator, MarkerCitationExtractor,
    answer_model, best_snippet, cited_numbers, format_sources,
)
from models import RetrievalResult
from test_models import make_chunk

CONTENTS = ["The estimator converges in ten steps. It uses a forgetting factor.",
            "Enrollment rates were imputed. Missing data was common."]


def results() -> list[RetrievalResult]:
    return [RetrievalResult(chunk=make_chunk(c, end_char=len(c), page_number=i + 1, end_page_number=i + 1,
                                             source=f"paper{i}.pdf"), score=0.9 - i / 10, retriever_type="hybrid")
            for i, c in enumerate(CONTENTS)]


def test_components_are_abstract():
    for cls in (Generator, CitationExtractor):
        with pytest.raises(TypeError):
            cls()


def test_cited_numbers_reads_single_grouped_and_repeated_markers():
    assert cited_numbers("A [2]. B [1, 3][2]. C [10].") == [2, 1, 3, 10]
    assert cited_numbers("no markers") == []


def test_format_sources_numbers_chunks_with_file_and_page():
    text = format_sources(results())
    assert text.startswith("[1] (paper0.pdf, page 1)\nThe estimator") and "\n\n[2] (paper1.pdf, page 2)" in text


def test_generated_answer_rejects_sources_that_do_not_exist():
    ok = {"answer": "Yes [1].", "citations": [{"source_number": 2, "quote": "x"}], "confidence": 0.5}
    assert answer_model(2).model_validate(ok).answer == "Yes [1]."
    assert answer_model(2) is answer_model(2) and answer_model(2).__name__ == "GeneratedAnswer"
    with pytest.raises(ValueError, match=r"\[3\]"):
        answer_model(2).model_validate({**ok, "answer": "Yes [3]."})
    assert GeneratedAnswer.model_validate({**ok, "answer": "Yes [3]."}).answer == "Yes [3]."   # unbound: no check
    with pytest.raises(ValueError):
        GeneratedAnswer.model_validate({**ok, "confidence": 1.2})


def test_best_snippet_keeps_real_quotes_and_replaces_invented_ones():
    content = CONTENTS[0]
    assert best_snippet(content, "the estimator CONVERGES in ten steps") == "the estimator CONVERGES in ten steps"
    assert best_snippet(content, "uses a forgetting factor of 0.98 always") == "It uses a forgetting factor."


def test_marker_extractor_builds_citations_from_the_retrieved_chunks():
    rs = results()
    citations = MarkerCitationExtractor().extract("Fast [2]. Also [1] and [9].", rs,
                                                  quotes={1: "It uses a forgetting factor"})
    assert [(c.source, c.page_number, round(c.relevance_score, 2)) for c in citations] == [
        ("paper1.pdf", 2, 0.8), ("paper0.pdf", 1, 0.9)]
    assert citations[1].text_snippet == "It uses a forgetting factor"
    assert citations[0].chunk_id == rs[1].chunk.id


def test_llm_generator_returns_a_validated_response_with_citations_and_usage():
    client = FakeLLMClient(cite_first_source)
    generator = LLMGenerator("vendor/model-x", client=client)
    response, usage = generator.generate_with_usage("How fast?", results())
    assert response.answer.startswith("vendor/model-x says") and response.confidence == 0.8
    assert [c.source for c in response.citations] == ["paper0.pdf"]
    assert response.citations[0].text_snippet == "The estimator converges in ten steps"
    assert len(response.chunks_used) == 2 and usage == {"prompt_tokens": 100, "completion_tokens": 20}
    call = client.calls[0]
    assert call["context"] is None and call["response_model"].n_sources == 2 and call["temperature"] == 0
    assert "Question: How fast?" in call["messages"][1]["content"]
    assert "cite" in call["messages"][0]["content"]


def test_llm_generator_retries_network_errors_and_skips_the_llm_without_sources():
    client = FakeLLMClient(cite_first_source, fail_times=1)
    generator = LLMGenerator("m", client=client, sleep=lambda s: None)
    assert generator.generate("q", results()).citations
    assert len(client.calls) == 2
    response = generator.generate("q", [])
    assert response.answer == NO_SOURCES_ANSWER and response.confidence == 0 and len(client.calls) == 2


def test_llm_judge_scores_an_answer_against_the_reference():
    response = LLMGenerator("m", client=FakeLLMClient(cite_first_source)).generate("How fast?", results())
    client = FakeLLMClient(perfect_scores)
    scores = LLMAnswerJudge("judge/model", client=client).judge(response, "Ten steps.")
    assert (scores.relevance, scores.citation_quality, scores.overall) == (5, 5, 4.5)
    prompt = client.calls[0]["messages"][1]["content"]
    assert "Reference answer: Ten steps." in prompt and "[2] (paper1.pdf, page 2)" in prompt


def test_prompts_with_template_braces_are_sent_unchanged():
    """LaTeX like {E}^{2} or {% must reach the model as-is (Instructor templating is never triggered)."""
    rs = results()
    latex = "Energy {E}^{2} and {% raw %} and }} appear."
    rs[0] = rs[0].model_copy(update={"chunk": make_chunk(latex, end_char=len(latex))})
    client = FakeLLMClient(cite_first_source)
    LLMGenerator("m", client=client).generate("What is {x}?", rs)
    prompt = client.calls[0]["messages"][1]["content"]
    assert latex in prompt and "What is {x}?" in prompt


def test_llm_judge_rejects_out_of_range_scores():
    bad = FakeLLMClient(lambda *a: {**perfect_scores(*a), "accuracy": 6})
    with pytest.raises(ValueError):
        LLMAnswerJudge("j", client=bad, max_attempts=1).judge(
            LLMGenerator("m", client=FakeLLMClient(cite_first_source)).generate("q", results()), "ref")
