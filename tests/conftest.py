"""Shared fixtures for the src/ test suite.

pytest.ini sets `pythonpath = src`, so every module in src/ is importable as a
top-level module here, matching how the project imports them when run from src/.
"""
import re
import zlib
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from embedding import Embedder
from parser import BasicPreprocessor, Heading, RawDocument, build_document
from models import Document

PROJECT_ROOT = Path(__file__).parent.parent
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
QRELS_FILE = PROJECT_ROOT / "data" / "pdf" / "arxiv" / "qrels.json"


class FakeEmbedder(Embedder):
    """Deterministic bag-of-words vectors: texts sharing words are similar. Token
    counts are word counts, so truncation is easy to trigger."""

    def __init__(self, name: str = "fake-model", dim: int = 32, max_tokens: Optional[int] = None) -> None:
        self.name, self._dim, self._max_tokens = name, dim, max_tokens
        self.calls = 0

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def max_tokens(self) -> Optional[int]:
        return self._max_tokens

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        vectors = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                vectors[row, zlib.crc32(word.encode()) % self._dim] += 1.0
            vectors[row, 0] += 1e-3                 # never all zeros
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def count_tokens(self, texts: list[str]) -> Optional[list[int]]:
        return [len(t.split()) for t in texts]


def make_document(pages: list[str], source: str = "0000.00001v1.pdf", toc: Optional[list[Heading]] = None,
                  has_qrels: bool = True) -> Document:
    """A Document built from page texts the same way the parser builds one."""
    raw = RawDocument(source=source, pages=pages, parser="pymupdf", title="Test Paper", toc_headings=toc or [])
    return build_document(raw, BasicPreprocessor(), has_qrels)


def write_pdf(path: Path, pages: list[list[tuple[str, float, str]]], toc: Optional[list[list]] = None,
              title: str = "", author: str = "") -> Path:
    """A PDF with one (text, font size, font name) line per entry; "hebo" is bold Helvetica."""
    import pymupdf

    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page()
        y = 72.0
        for text, size, font in lines:
            page.insert_text((72, y), text, fontsize=size, fontname=font)
            y += size * 2.2
    if toc:
        doc.set_toc(toc)
    doc.set_metadata({"title": title, "author": author})
    doc.save(path)
    doc.close()
    return path


BODY = "helv"
BOLD = "hebo"


def paper_pages(topic: str = "impedance") -> list[list[tuple[str, float, str]]]:
    """Two pages of a small paper with a title, numbered bold headings, and body text."""
    body = [f"The {topic} estimate improves with more data.", f"We measure {topic} on a test grid.",
            "Results are reported for every case.", "Each case uses the same settings."]
    return [
        [("A Study of Grid Estimation", 20, BODY), ("1. Introduction", 13, BOLD), *[(s, 10, BODY) for s in body],
         ("2. Methods", 13, BOLD), *[(s, 10, BODY) for s in body]],
        [("3. Results", 13, BOLD), *[(s, 10, BODY) for s in body], ("4. Conclusion", 13, BOLD),
         ("The method works well in practice.", 10, BODY)],
    ]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def papers_dir(tmp_path) -> Path:
    """Three generated papers; the qrels fixture marks two of them as having queries."""
    d = tmp_path / "papers"
    d.mkdir()
    write_pdf(d / "2401.00001v1.pdf", paper_pages("impedance"), toc=[[1, "Introduction", 1], [1, "Methods", 1],
                                                                      [1, "Results", 2], [1, "Conclusion", 2]])
    write_pdf(d / "2401.00002v1.pdf", paper_pages("enrollment"), title="Enrollment Paper", author="A. Author")
    write_pdf(d / "2401.00003v1.pdf", paper_pages("particles"))
    return d


@pytest.fixture
def qrels_file(tmp_path) -> Path:
    path = tmp_path / "qrels.json"
    path.write_text('{"q1": {"doc_id": "2401.00001v1", "section_id": 1},'
                    ' "q2": {"doc_id": "2401.00002v1", "section_id": 0}}')
    return path


class FakeLLMClient:
    """Stands in for an Instructor client. `respond(model, messages, response_model)` returns
    the raw dict the "LLM" produced; it's validated like Instructor would (with `context`),
    and `fail_times` calls raise first. Calls are recorded."""

    class Usage:
        prompt_tokens, completion_tokens = 100, 20

    class Completion:
        def __init__(self) -> None:
            self.usage = FakeLLMClient.Usage()

    def __init__(self, respond, fail_times: int = 0, error: Exception = None) -> None:
        self.respond, self.fail_times, self.error = respond, fail_times, error or ConnectionError("network down")
        self.calls: list[dict] = []
        self.chat = self
        self.completions = self

    def create_with_completion(self, model, messages, response_model, max_retries=0, context=None, **kwargs):
        self.calls.append({"model": model, "messages": messages, "context": context, "response_model": response_model,
                           **kwargs})
        if self.fail_times:
            self.fail_times -= 1
            raise self.error
        raw = self.respond(model, messages, response_model)
        return response_model.model_validate(raw, context=context), FakeLLMClient.Completion()

    def create(self, model, messages, response_model, max_retries=0, context=None, **kwargs):
        return self.create_with_completion(model, messages, response_model, max_retries, context, **kwargs)[0]


def cite_first_source(model, messages, response_model) -> dict:
    """A generator reply citing source [1] with a quote taken from it."""
    user = messages[-1]["content"]
    first = user.split("[1] (", 1)[1].split("\n", 1)[1].split("\n\n[2]", 1)[0]
    return {"answer": f"{model} says the answer is in the first source [1].",
            "citations": [{"source_number": 1, "quote": first.split(".")[0]}], "confidence": 0.8}


def perfect_scores(model, messages, response_model) -> dict:
    return {"relevance": 5, "accuracy": 4, "completeness": 4, "citation_quality": 5, "reasoning": "Fine."}
