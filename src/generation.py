"""
Query phase step 3: generate an answer with citations from the retrieved chunks.

The retrieved chunks are numbered [1]..[k] in the prompt (templates/generation_*.txt).
Through Instructor (tools mode: the provider's function calling builds the JSON)
the LLM returns a GeneratedAnswer: the answer text with [n] markers, a verbatim
quote per cited source, and a confidence. Instructor re-asks the model when the
output doesn't validate (e.g. it cites [7] of 5 sources). The source count is bound
to the response model (answer_model) rather than passed as Instructor `context`,
which would make Instructor render the prompt as a Jinja template — and chunk text
full of LaTeX braces breaks that.

The LLM never writes chunk ids, file names, page numbers, or scores: the citation
extractor builds each Citation from the retrieved chunk the [n] points to. A quote
is kept as the citation's text_snippet only when it actually occurs in that chunk;
otherwise the chunk sentence sharing the most words with it is used.
"""

import functools
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, ClassVar, Optional

from pydantic import BaseModel, Field, model_validator

from api_retry import call_with_retries
from chunking import sentence_spans
from config import GENERATION_TEMPERATURE, LLM_MAX_ATTEMPTS, PROJECT_ROOT
from models import Citation, QAResponse, RetrievalResult

TEMPLATES_DIR = PROJECT_ROOT / "templates"
# Instructor re-asks this many times when the output fails validation.
VALIDATION_RETRIES = 2
MAX_SNIPPET_CHARS = 300
NO_SOURCES_ANSWER = "No relevant sources were retrieved, so the question can't be answered."

_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_WORD_RE = re.compile(r"\w+")


def load_template(name: str, templates_dir: Path = TEMPLATES_DIR) -> str:
    return (templates_dir / name).read_text(encoding="utf-8")


def cited_numbers(text: str) -> list[int]:
    """Source numbers in [n] / [n, m] markers, in order of first appearance."""
    seen: dict[int, None] = {}
    for m in _MARKER_RE.finditer(text):
        for n in m.group(1).split(","):
            seen.setdefault(int(n), None)
    return list(seen)


# ── LLM output schema ─────────────────────────────────────────────────────────

class CitedSource(BaseModel):
    source_number: int = Field(ge=1, description="The [n] number of a source the answer cites")
    quote: str = Field(description="A short verbatim quote from that source supporting the answer")


class GeneratedAnswer(BaseModel):
    # How many sources the prompt numbered; set per call by answer_model(). None skips the check.
    n_sources: ClassVar[Optional[int]] = None

    answer: str = Field(description="The answer, with [n] markers after each claim")
    citations: list[CitedSource] = Field(default_factory=list,
                                         description="One entry per cited source, with a supporting quote")
    confidence: float = Field(ge=0.0, le=1.0, description="Probability (0-1) the answer is correct and supported")

    @model_validator(mode="after")
    def _sources_exist(self) -> "GeneratedAnswer":
        n_sources = type(self).n_sources
        if n_sources is not None:
            numbers = cited_numbers(self.answer) + [c.source_number for c in self.citations]
            bad = sorted({n for n in numbers if not 1 <= n <= n_sources})
            if bad:
                raise ValueError(f"Cited source(s) {bad} don't exist; only [1]..[{n_sources}] were given")
        return self


@functools.cache
def answer_model(n_sources: int) -> type[GeneratedAnswer]:
    """GeneratedAnswer that only accepts citations of sources [1]..[n_sources]."""
    return type("GeneratedAnswer", (GeneratedAnswer,), {"n_sources": n_sources, "__module__": __name__})


def format_sources(results: list[RetrievalResult], max_chars: Optional[int] = None) -> str:
    """The retrieved chunks numbered [1]..[k], each with its file, page, and section."""
    blocks = []
    for i, r in enumerate(results, 1):
        m = r.chunk.metadata
        section = f', section "{m.section_title}"' if m.section_title else ""
        content = r.chunk.content if max_chars is None else r.chunk.content[:max_chars]
        blocks.append(f"[{i}] ({m.source}, page {m.page_number}{section})\n{content}")
    return "\n\n".join(blocks)


# ── Citation extraction ───────────────────────────────────────────────────────

class CitationExtractor(ABC):
    @abstractmethod
    def extract(self, answer: str, results: list[RetrievalResult],
                quotes: Optional[dict[int, str]] = None) -> list[Citation]:
        """Citations for the sources the answer cites. `quotes` maps a source number to
        the LLM's supporting quote from it, when it gave one."""


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def best_snippet(content: str, quote: str) -> str:
    """`quote` when it occurs in `content` (ignoring case, spacing, and punctuation),
    else the sentence of `content` sharing the most words with it."""
    if quote and _normalize(quote) in _normalize(content):
        return quote.strip()[:MAX_SNIPPET_CHARS]
    words = set(_WORD_RE.findall(quote.lower()))
    sentences = [content[s:e] for s, e in sentence_spans(content)] or [content]
    best = max(sentences, key=lambda s: len(words & set(_WORD_RE.findall(s.lower()))))
    return best[:MAX_SNIPPET_CHARS]


class MarkerCitationExtractor(CitationExtractor):
    """One Citation per source number: those in the answer's [n] markers, then any
    others the LLM listed. Numbers outside 1..len(results) are ignored."""

    def extract(self, answer: str, results: list[RetrievalResult],
                quotes: Optional[dict[int, str]] = None) -> list[Citation]:
        quotes = quotes or {}
        numbers = list(dict.fromkeys([*cited_numbers(answer), *quotes]))
        citations = []
        for n in numbers:
            if not 1 <= n <= len(results):
                continue
            r = results[n - 1]
            citations.append(Citation(chunk_id=r.chunk.id, source=r.chunk.metadata.source,
                                      page_number=r.chunk.metadata.page_number,
                                      text_snippet=best_snippet(r.chunk.content, quotes.get(n, "")),
                                      relevance_score=r.score))
        return citations


# ── Generators ────────────────────────────────────────────────────────────────

class Generator(ABC):
    name: str

    @abstractmethod
    def generate(self, query: str, results: list[RetrievalResult]) -> QAResponse:
        """An answer to `query` grounded in the retrieved chunks."""

    def generate_with_usage(self, query: str, results: list[RetrievalResult]) -> tuple[QAResponse, dict]:
        """The response and the call's token usage; generators that can't report usage return {}."""
        return self.generate(query, results), {}


class LLMGenerator(Generator):
    """Any OpenRouter chat model through Instructor. `client` is an Instructor client
    (config.make_client() when omitted). Safe to call from several threads."""

    def __init__(self, model: str, client=None, temperature: float = GENERATION_TEMPERATURE,
                 extractor: Optional[CitationExtractor] = None, max_attempts: int = LLM_MAX_ATTEMPTS,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if client is None:
            from config import make_client
            client = make_client()
        self.name, self.model, self.client, self.temperature = model, model, client, temperature
        self.extractor = extractor or MarkerCitationExtractor()
        self.max_attempts, self._sleep = max_attempts, sleep
        self.system_prompt = load_template("generation_system.txt")
        self.user_template = load_template("generation_user.txt")

    def generate(self, query: str, results: list[RetrievalResult]) -> QAResponse:
        return self.generate_with_usage(query, results)[0]

    def generate_with_usage(self, query: str, results: list[RetrievalResult]) -> tuple[QAResponse, dict]:
        """The response and the call's token usage ({"prompt_tokens", "completion_tokens"})."""
        if not results:
            return QAResponse(query=query, answer=NO_SOURCES_ANSWER, confidence=0.0), {}
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.user_template.format(query=query, sources=format_sources(results))}]

        def call():
            return self.client.chat.completions.create_with_completion(
                model=self.model, messages=messages, response_model=answer_model(len(results)),
                max_retries=VALIDATION_RETRIES, temperature=self.temperature)

        generated, completion = call_with_retries(call, self.max_attempts, sleep=self._sleep)
        usage = getattr(completion, "usage", None)
        quotes = {c.source_number: c.quote for c in generated.citations}
        response = QAResponse(query=query, answer=generated.answer,
                              citations=self.extractor.extract(generated.answer, results, quotes),
                              chunks_used=[r.chunk for r in results], confidence=generated.confidence)
        return response, {"prompt_tokens": getattr(usage, "prompt_tokens", None),
                          "completion_tokens": getattr(usage, "completion_tokens", None)}


def make_generator(model: str, client=None) -> Generator:
    return LLMGenerator(model, client)
