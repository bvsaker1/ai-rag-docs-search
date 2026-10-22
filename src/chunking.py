"""
Ingestion step 2: split each Document into Chunks with one chunking method.

Chunks never cross documents. They may cross pages; FIXED and SEMANTIC chunks may
also cross sections, RECURSIVE chunks never do. Each chunk records the page it
starts and ends on and the section it starts in. Sizes are tiktoken
tokens (TIKTOKEN_ENCODING); start_char/end_char are offsets into Document.content.

Methods:
    FIXED     sliding window of CHUNK_SIZE tokens, OVERLAP tokens of overlap
    RECURSIVE each section (sections under RECURSIVE_MIN_SECTION_TOKENS merge into
              the next) is split by paragraphs, then sentences, then words, then
              tokens — each level only for pieces still over CHUNK_SIZE tokens. The
              pieces are packed back up to CHUNK_SIZE tokens per chunk; the next chunk
              repeats the trailing pieces that fit in OVERLAP tokens
    SEMANTIC  sentences embedded with SEMANTIC_CHUNK_EMBEDDING_MODEL; cut where
              neighboring sentences' cosine distance is at or above the document's
              SEMANTIC_BREAKPOINT_PERCENTILE percentile, once a chunk has
              SEMANTIC_MIN_TOKENS tokens; never over CHUNK_SIZE tokens; no overlap

SEMANTIC sentences never cross a paragraph or page break; a sentence longer than
CHUNK_SIZE tokens is split into CHUNK_SIZE-token pieces.

Chunk ids are deterministic (uuid5 of config + source + index), so re-chunking
with the same settings yields the same ids.
"""

import bisect
import functools
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import (
    CHUNK_METHOD_NAMES, RECURSIVE_MIN_SECTION_TOKENS, SEMANTIC_BREAKPOINT_PERCENTILE, SEMANTIC_CHUNK_EMBEDDING_MODEL,
    SEMANTIC_MIN_TOKENS, TIKTOKEN_ENCODING,
)
from embedding import Embedder
from models import Chunk, ChunkingConfig, ChunkMetadata, Document

# A sentence ends at . ! or ? (plus closing quotes/brackets) before whitespace and a capital, digit, or bracket.
_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*(?=\s+[A-Z0-9\"(\[])")
# Tokens ending in "." that don't end a sentence ("et al. Smith", "Fig. 3", "Eq. 2").
_ABBREVIATIONS = {"al", "e.g", "i.e", "etc", "cf", "vs", "fig", "figs", "eq", "eqs", "ref", "refs", "sec", "secs",
                  "no", "nos", "approx", "resp", "dr", "mr", "mrs", "ms", "prof", "st", "vol", "pp", "ch", "tab",
                  "thm", "def", "prop", "lem", "cor", "alg", "appx", "jan", "feb", "mar", "apr", "jun", "jul",
                  "aug", "sep", "sept", "oct", "nov", "dec"}
_PARAGRAPH_RE = re.compile(r"[^\n]+(?:\n(?!\n)[^\n]+)*")
_WORD_RE = re.compile(r"\S+")
# RECURSIVE splits each section at these boundaries, in order, then into raw token windows.
RECURSIVE_SEPARATORS = ["paragraph", "sentence", "word"]


@functools.cache
def get_encoding(name: str = TIKTOKEN_ENCODING):
    import tiktoken
    return tiktoken.get_encoding(name)


def count_tokens(text: str, encoding: str = TIKTOKEN_ENCODING) -> int:
    return len(get_encoding(encoding).encode(text, disallowed_special=()))


def token_spans(text: str, encoding: str = TIKTOKEN_ENCODING) -> list[tuple[int, int]]:
    """(start, end) character offsets of every token in `text`."""
    enc = get_encoding(encoding)
    tokens = enc.encode(text, disallowed_special=())
    if not tokens:
        return []
    _, starts = enc.decode_with_offsets(tokens)
    ends = starts[1:] + [len(text)]
    return list(zip(starts, ends))


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink [start, end) past leading and trailing whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


# ── Sentences ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Unit:
    """A sentence (or a piece of an over-long one) as a document span."""
    start: int
    end: int
    tokens: int


def _is_abbreviation(text: str, period_index: int) -> bool:
    before = text[:period_index]
    token = before.rsplit(None, 1)[-1] if before.strip() else ""
    token = token.lower().lstrip("([\"'")
    return token in _ABBREVIATIONS or (len(token) == 1 and token.isalpha())


def sentence_spans(text: str, offset: int = 0) -> list[tuple[int, int]]:
    """Sentence (start, end) spans in `text`, shifted by `offset`, whitespace trimmed.
    Paragraph breaks (blank lines) always end a sentence."""
    spans = []
    for para in _PARAGRAPH_RE.finditer(text):
        p_text, p_start, start = para.group(0), para.start(), 0
        for m in _SENTENCE_END_RE.finditer(p_text):
            if p_text[m.start()] == "." and _is_abbreviation(p_text, m.start()):
                continue
            spans.append((p_start + start, p_start + m.end()))
            start = m.end()
        spans.append((p_start + start, p_start + len(p_text)))
    trimmed = [_trim(text, s, e) for s, e in spans]
    return [(offset + s, offset + e) for s, e in trimmed if e > s]


def build_units(document: Document, max_tokens: int, encoding: str = TIKTOKEN_ENCODING) -> list[Unit]:
    """The document's sentences, page by page; sentences over `max_tokens` are split."""
    content, starts = document.content, document.metadata.page_starts
    bounds = starts + [len(content)]
    units: list[Unit] = []
    for i in range(len(starts)):
        for s, e in sentence_spans(content[bounds[i]:bounds[i + 1]], bounds[i]):
            spans = token_spans(content[s:e], encoding)
            for first in range(0, len(spans), max_tokens):
                piece = spans[first:first + max_tokens]
                ps, pe = _trim(content, s + piece[0][0], s + piece[-1][1])
                if pe > ps:
                    units.append(Unit(ps, pe, len(piece)))
    return units


# ── Grouping ──────────────────────────────────────────────────────────────────

def pack_groups(units: list[Unit], max_tokens: int, overlap_tokens: int) -> list[tuple[int, int]]:
    """Greedy packing of whole units into (first, last+1) ranges. Each next group
    starts with the previous group's trailing units that fit in `overlap_tokens`.
    Overlap never repeats a whole group and never leaves the next group without
    room for a new unit."""
    groups: list[tuple[int, int]] = []
    i = 0
    while i < len(units):
        j, tokens = i, 0
        while j < len(units) and (j == i or tokens + units[j].tokens <= max_tokens):
            tokens += units[j].tokens
            j += 1
        groups.append((i, j))
        if j == len(units):
            break
        k, carried = j, 0
        while k - 1 > i:
            candidate = carried + units[k - 1].tokens
            if candidate > overlap_tokens or candidate + units[j].tokens > max_tokens:
                break
            k -= 1
            carried = candidate
        i = k
    return groups


def semantic_groups(units: list[Unit], distances: np.ndarray, percentile: float, min_tokens: int,
                    max_tokens: int) -> list[tuple[int, int]]:
    """Cut after unit i when distances[i] (to unit i+1) is a breakpoint and the
    group has `min_tokens`; always cut before exceeding `max_tokens`. A final group
    under `min_tokens` merges into the previous one when it fits."""
    if not units:
        return []
    threshold = np.percentile(distances, percentile) if len(distances) else np.inf
    groups: list[tuple[int, int]] = []
    start, tokens = 0, units[0].tokens
    for i in range(1, len(units)):
        breakpoint_here = distances[i - 1] >= threshold and tokens >= min_tokens
        if breakpoint_here or tokens + units[i].tokens > max_tokens:
            groups.append((start, i))
            start, tokens = i, 0
        tokens += units[i].tokens
    groups.append((start, len(units)))
    if len(groups) > 1 and tokens < min_tokens:
        prev_start, _ = groups[-2]
        if sum(u.tokens for u in units[prev_start:]) <= max_tokens:
            groups[-2:] = [(prev_start, len(units))]
    return groups


# ── Chunkers ──────────────────────────────────────────────────────────────────

class Chunker(ABC):
    """Turns a Document into Chunks. Subclasses only decide the character spans;
    building Chunk records (ids, pages, sections, token counts) is shared."""

    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    @abstractmethod
    def spans(self, document: Document) -> list[tuple[int, int, int]]:
        """(start_char, end_char, tokens) of every chunk, in document order. `tokens`
        is the count the chunker packed, which is what chunk_size limits."""

    def chunk(self, document: Document) -> list[Chunk]:
        return [self._make_chunk(document, i, s, e, n) for i, (s, e, n) in enumerate(self.spans(document))]

    def _make_chunk(self, document: Document, index: int, start: int, end: int, tokens: int) -> Chunk:
        meta = document.metadata
        content = document.content[start:end]
        section = None
        if meta.sections:
            section_starts = [s.start_char for s in meta.sections]
            section = meta.sections[max(bisect.bisect_right(section_starts, start) - 1, 0)]
        return Chunk(
            id=Chunk.make_id(self.config, meta.source, index),
            content=content,
            metadata=ChunkMetadata(
                document_id=document.id, source=meta.source, arxiv_id=meta.arxiv_id,
                page_number=bisect.bisect_right(meta.page_starts, start),
                end_page_number=bisect.bisect_right(meta.page_starts, end - 1),
                start_char=start, end_char=end, chunk_index=index,
                section_index=section.index if section else None, section_title=section.title if section else None,
                method=self.config.chunker, token_count=max(tokens, 1),
            ),
        )


class FixedTokenChunker(Chunker):
    """Sliding window of `chunk_size` tokens over the whole document, advancing by
    `chunk_size - overlap`, stopping at the first window that reaches the end."""

    def spans(self, document: Document) -> list[tuple[int, int, int]]:
        text, size = document.content, self.config.chunk_size
        tokens = token_spans(text, self.config.tokenizer)
        step = size - self.config.overlap
        spans = []
        for first in range(0, len(tokens), step):
            window = tokens[first:first + size]
            s, e = _trim(text, window[0][0], window[-1][1])
            if e > s:
                spans.append((s, e, len(window)))
            if first + size >= len(tokens):
                break
        return spans


def _group_spans(units: list[Unit], groups: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    return [(units[a].start, units[b - 1].end, sum(u.tokens for u in units[a:b])) for a, b in groups]


def _separate(content: str, start: int, end: int, separator: str) -> list[tuple[int, int]]:
    """Trimmed spans of content[start:end] split at `separator` boundaries."""
    text = content[start:end]
    if separator == "sentence":
        return sentence_spans(text, start)
    pattern = _PARAGRAPH_RE if separator == "paragraph" else _WORD_RE
    spans = [_trim(text, m.start(), m.end()) for m in pattern.finditer(text)]
    return [(start + s, start + e) for s, e in spans if e > s]


def recursive_units(content: str, start: int, end: int, max_tokens: int, encoding: str = TIKTOKEN_ENCODING,
                    level: int = 0) -> list[Unit]:
    """content[start:end] as pieces of at most `max_tokens` tokens: whole when it fits,
    else split at RECURSIVE_SEPARATORS[level] and each piece recursed on, and past the
    last separator cut into `max_tokens`-token windows."""
    tokens = count_tokens(content[start:end], encoding)
    if tokens <= max_tokens:
        return [Unit(start, end, tokens)] if tokens else []
    if level == len(RECURSIVE_SEPARATORS):
        spans = token_spans(content[start:end], encoding)
        units = []
        for first in range(0, len(spans), max_tokens):
            piece = spans[first:first + max_tokens]
            s, e = _trim(content, start + piece[0][0], start + piece[-1][1])
            if e > s:
                units.append(Unit(s, e, len(piece)))
        return units
    pieces = _separate(content, start, end, RECURSIVE_SEPARATORS[level])
    return [u for s, e in pieces for u in recursive_units(content, s, e, max_tokens, encoding, level + 1)]


def merged_section_spans(document: Document, min_tokens: int, encoding: str = TIKTOKEN_ENCODING) -> list[tuple[int, int]]:
    """Section spans, with any section under `min_tokens` tokens merged into the next
    (a parent heading directly followed by its first subheading). No sections: the whole document."""
    sections = document.metadata.sections
    if not sections:
        return [(0, len(document.content))]
    spans: list[tuple[int, int]] = []
    start = sections[0].start_char
    for i, section in enumerate(sections):
        last = i == len(sections) - 1
        if last or count_tokens(document.content[start:section.end_char], encoding) >= min_tokens:
            spans.append((start, section.end_char))
            if not last:
                start = sections[i + 1].start_char
    return spans


class RecursiveChunker(Chunker):
    def spans(self, document: Document) -> list[tuple[int, int, int]]:
        c = self.config
        spans: list[tuple[int, int, int]] = []
        for start, end in merged_section_spans(document, c.min_tokens or 0, c.tokenizer):
            units = recursive_units(document.content, start, end, c.chunk_size, c.tokenizer)
            spans += _group_spans(units, pack_groups(units, c.chunk_size, c.overlap))
        return spans


class SemanticChunker(Chunker):
    def __init__(self, config: ChunkingConfig, embedder: Embedder) -> None:
        super().__init__(config)
        self.embedder = embedder

    def spans(self, document: Document) -> list[tuple[int, int, int]]:
        c = self.config
        units = build_units(document, c.chunk_size, c.tokenizer)
        if not units:
            return []
        vectors = self.embedder.embed([document.content[u.start:u.end] for u in units])
        distances = 1 - np.sum(vectors[:-1] * vectors[1:], axis=1)
        groups = semantic_groups(units, distances, c.breakpoint_percentile, c.min_tokens, c.chunk_size)
        return _group_spans(units, groups)


CHUNKERS: dict[str, type[Chunker]] = {
    "fixed_token": FixedTokenChunker,
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
}


def build_config(method: str, chunk_size: int, overlap: int, parser: str, tokenizer: str = TIKTOKEN_ENCODING,
                 breakpoint_percentile: float = SEMANTIC_BREAKPOINT_PERCENTILE,
                 min_tokens: int = SEMANTIC_MIN_TOKENS) -> ChunkingConfig:
    """Map a .env method name (FIXED/RECURSIVE/SEMANTIC) to a ChunkingConfig."""
    chunker = CHUNK_METHOD_NAMES.get(method.upper())
    if chunker is None:
        raise ValueError(f"Unknown chunking method {method!r}; expected one of {', '.join(CHUNK_METHOD_NAMES)}")
    common = dict(parser=parser, chunker=chunker, chunk_size=chunk_size, tokenizer=tokenizer)
    if chunker == "recursive":
        return ChunkingConfig(**common, overlap=overlap, separators=["section", *RECURSIVE_SEPARATORS, "token"],
                              min_tokens=RECURSIVE_MIN_SECTION_TOKENS)
    if chunker == "semantic":
        return ChunkingConfig(**common, overlap=0, boundary_embedding_model=SEMANTIC_CHUNK_EMBEDDING_MODEL,
                              breakpoint_percentile=breakpoint_percentile, min_tokens=min_tokens)
    return ChunkingConfig(**common, overlap=overlap)


def make_chunker(config: ChunkingConfig, boundary_embedder: Optional[Embedder] = None) -> Chunker:
    """The Chunker for `config`; semantic chunking needs `boundary_embedder`."""
    cls = CHUNKERS.get(config.chunker)
    if cls is None:
        raise ValueError(f"Unknown chunker {config.chunker!r}; expected one of {', '.join(CHUNKERS)}")
    if cls is SemanticChunker:
        if boundary_embedder is None:
            raise ValueError("Semantic chunking needs a boundary embedder")
        return SemanticChunker(config, boundary_embedder)
    return cls(config)


def chunk_documents(documents: list[Document], chunker: Chunker) -> list[Chunk]:
    """Every document's chunks; chunk_index restarts at 0 for each document."""
    return [chunk for document in documents for chunk in chunker.chunk(document)]
