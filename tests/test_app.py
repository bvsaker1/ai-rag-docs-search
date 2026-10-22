"""Smoke tests of the Streamlit app (src/app.py) with Streamlit's AppTest: the real script runs,
with the corpus retriever and the LLM replaced by fakes."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import generation
import session_index
from conftest import FakeLLMClient, cite_first_source
from models import RetrieverType
from retrieval import Retriever, _results
from test_models import make_chunk

APP = str(Path(__file__).parent.parent / "src" / "app.py")
TEXTS = ["The estimator converges in ten steps. It uses a forgetting factor $\\lambda$.",
         "Enrollment rates were imputed with a Bayesian model."]


class FakeCorpus(Retriever):
    retriever_type = RetrieverType.HYBRID

    def __init__(self):
        self.chunks = [make_chunk(t, end_char=len(t), source=f"paper{i}.pdf") for i, t in enumerate(TEXTS)]

    def retrieve(self, query, k):
        return _results(self.chunks, [(i, 1.0 - i / 10) for i in range(min(k, len(self.chunks)))],
                        self.retriever_type)


@pytest.fixture
def app(monkeypatch):
    st.cache_resource.clear()
    real_generator = generation.LLMGenerator
    monkeypatch.setattr(generation, "LLMGenerator",
                        lambda model: real_generator(model, client=FakeLLMClient(cite_first_source)))
    monkeypatch.setattr(session_index, "corpus_retriever", lambda s, config: FakeCorpus())
    monkeypatch.setattr(session_index, "make_reranker", lambda name: None)
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    return at


def test_app_renders_the_configuration_sidebar(app):
    assert not app.exception
    labels = [w.label for w in app.sidebar.selectbox]
    assert labels == ["Chunking", "Retrieval", "Embedding model", "Reranker", "Generator"]
    assert app.sidebar.slider[0].label == "Top-k chunks"
    assert app.sidebar.radio[0].value == "Corpus"                      # no uploads yet
    assert app.text_input[0].label == "Question"


def test_asking_shows_the_answer_citations_and_source_chunks(app):
    app.text_input[0].input("How fast does the estimator converge?")
    app.button[0].click()
    app.run()
    assert not app.exception and not app.error
    assert any("says the answer is in the first source [1]" in m.value for m in app.markdown)
    assert any("paper0.pdf, page 1" in m.value for m in app.markdown)            # citation line
    labels = [e.label for e in app.expander]
    assert len(labels) == 2 and labels[0].startswith("✅ #1 paper0.pdf")
    chunk_html = next(m.value for m in app.markdown if "<mark>" in m.value)
    assert "<mark>The estimator converges in ten steps</mark>" in chunk_html and "&#36;" in chunk_html


def test_searching_uploads_without_any_shows_an_error(app):
    app.sidebar.radio[0].set_value("Uploaded files")
    app.text_input[0].input("anything")
    app.button[0].click()
    app.run()
    assert app.error and "Upload at least one PDF" in app.error[0].value
