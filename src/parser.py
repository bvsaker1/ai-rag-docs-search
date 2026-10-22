"""
Ingestion step 1: pick the papers, parse each PDF, clean its text, and find its sections.

    select_papers → Parser.parse (one RawDocument per PDF) → Preprocessor.clean (per page)
      → build_document (pages joined, sections located) → Document

Sections are literal sections of the paper, found from the first source that
locates at least two headings in the cleaned text:
    toc   the PDF's bookmark outline (most LaTeX/arXiv papers have one)
    font  lines set larger than the body text, or short bold / numbered lines
          ("3.2 Model", "II. GRID MODELING")
    page  no headings found: one section per page
A heading that can't be found in the text is skipped. Text before the first
heading (title, authors, abstract) becomes a "Front matter" section.

A PDF that fails to open or has no text is logged and skipped; the rest still parse.
"""

import json
import re
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import Logger
from models import PAGE_SEPARATOR, Document, DocumentMetadata, Section

FRONT_MATTER_TITLE = "Front matter"
MIN_LOCATED_HEADINGS = 2
MAX_HEADING_WORDS = 15
MAX_TITLE_CHARS = 300
# Font headings: at least this many points above the body text size.
MIN_HEADING_SIZE_DELTA = 1.0
# Lines repeated on at least this many pages are running headers, not headings.
RUNNING_HEADER_MIN_PAGES = 3

# What may sit between a line start and a heading's title: nothing, or a section
# number like "3.2.1.", "II.", "B.", "2)", "Section 4", "Appendix A".
_HEADING_PREFIX_RE = re.compile(r"^(?:(?:section|appendix|chapter)\s+)?(?:\d+(?:\.\d+)*\.?|[IVXL]+\.|[A-Z]\.?|\d+\))?\s*$",
                                re.IGNORECASE)
# A section number directly before a heading found mid-line ("... ESTIMATION A. Thevenin").
_TRAILING_NUMBER_RE = re.compile(r"(?<!\w)(?:\d+(?:\.\d+)*\.?|[IVXL]+\.|[A-Z]\.|\d+\))\s+$")
_NUMBERED_HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*\.|[IVXL]+\.|[A-H]\.|\d+\))\s+[A-Z]")
# The arXiv identifier stamped down the margin of every arXiv PDF.
_ARXIV_STAMP_RE = re.compile(r"^arXiv:\d", re.IGNORECASE)
_CAPTION_RE = re.compile(r"^(?:fig(?:ure)?|table|algorithm|eq(?:uation)?)\.?\s*\d", re.IGNORECASE)
_NAMED_HEADINGS = {"abstract", "introduction", "related work", "background", "method", "methods", "methodology",
                   "experiments", "results", "discussion", "conclusion", "conclusions", "references",
                   "acknowledgments", "acknowledgements", "appendix", "bibliography"}
_BOLD_FLAG = 16


@dataclass(frozen=True)
class Heading:
    level: int
    title: str
    page_number: int


@dataclass
class RawDocument:
    """What a Parser extracts from one PDF, before cleaning. `toc_headings` and
    `font_headings` are candidate section headings from each source."""
    source: str
    pages: list[str]
    parser: str
    title: Optional[str] = None
    author: Optional[str] = None
    toc_headings: list[Heading] = field(default_factory=list)
    font_headings: list[Heading] = field(default_factory=list)

    @property
    def arxiv_id(self) -> str:
        return Path(self.source).stem


class EmptyDocumentError(ValueError):
    pass


# ── Paper selection ───────────────────────────────────────────────────────────

def qrels_doc_ids(qrels_path: Path) -> set[str]:
    """arXiv ids of every paper that some query's ground truth points into."""
    with open(qrels_path, encoding="utf-8") as f:
        return {entry["doc_id"] for entry in json.load(f).values()}


def select_papers(papers_dir: Path, qrel_ids: set[str], max_papers: int, max_negatives: int = 0) -> list[Path]:
    """The first `max_papers` PDFs (sorted by arXiv id) with qrels, then the first
    `max_negatives` without (hard negatives)."""
    pdfs = sorted(papers_dir.glob("*.pdf"), key=lambda p: p.stem)
    with_qrels = [p for p in pdfs if p.stem in qrel_ids][:max_papers]
    negatives = [p for p in pdfs if p.stem not in qrel_ids][:max_negatives]
    return with_qrels + negatives


# ── Parsers ───────────────────────────────────────────────────────────────────

class Parser(ABC):
    name: str

    @abstractmethod
    def parse(self, path: Path) -> RawDocument:
        """Extract per-page text, metadata, and candidate headings from one PDF."""


@dataclass(frozen=True)
class _Line:
    text: str
    size: float
    bold: bool
    page_number: int
    block_lines: int        # lines in the line's block
    block_words: int
    horizontal: bool = True


class PyMuPdfParser(Parser):
    """Page text is built from PyMuPDF's text blocks: lines joined with newlines,
    blocks with blank lines, in the PDF's content order (which keeps two-column
    papers in reading order, unlike sorting by position)."""

    name = "pymupdf"

    def parse(self, path: Path) -> RawDocument:
        import pymupdf

        with pymupdf.open(path) as doc:
            pages, lines = [], []
            for page in doc:
                text, page_lines = self._page(page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT), page.number + 1)
                pages.append(text)
                lines += page_lines
            meta = doc.metadata or {}
            toc = doc.get_toc(simple=True)
        return RawDocument(
            source=path.name,
            pages=pages,
            parser=self.name,
            title=_clean_meta(meta.get("title")) or largest_text_on_first_page(lines),
            author=_clean_meta(meta.get("author")),
            toc_headings=toc_headings(toc),
            font_headings=font_headings(lines),
        )

    @staticmethod
    def _page(page_dict: dict, page_number: int) -> tuple[str, list[_Line]]:
        blocks, lines = [], []
        for block in page_dict["blocks"]:
            block_lines = [(("".join(s["text"] for s in line["spans"])).strip(), line["spans"])
                           for line in block.get("lines", [])]
            block_lines = [(text, spans) for text, spans in block_lines if text]
            if not block_lines:
                continue
            blocks.append("\n".join(text for text, _ in block_lines))
            words = sum(len(text.split()) for text, _ in block_lines)
            horizontal = block["lines"][0]["dir"] == (1.0, 0.0)
            for text, spans in block_lines:
                visible = [s for s in spans if s["text"].strip()]
                lines.append(_Line(
                    text=text,
                    size=max(s["size"] for s in visible),
                    bold=all(s["flags"] & _BOLD_FLAG for s in visible),
                    page_number=page_number,
                    block_lines=len(block_lines),
                    block_words=words,
                    horizontal=horizontal,
                ))
        return "\n\n".join(blocks), lines


def _clean_meta(value: Optional[str]) -> Optional[str]:
    value = " ".join((value or "").split())
    return value or None


def toc_headings(toc: list[list]) -> list[Heading]:
    """Bookmark entries with a title and a resolved page."""
    return [Heading(level, " ".join(title.split()), page) for level, title, page, *_ in toc
            if title.strip() and page >= 1]


def body_size(lines: list[_Line]) -> float:
    """The font size covering the most characters."""
    counts = Counter()
    for line in lines:
        counts[round(line.size, 1)] += len(line.text)
    return counts.most_common(1)[0][0] if counts else 0.0


def largest_text_on_first_page(lines: list[_Line]) -> Optional[str]:
    """The paper's title when the PDF metadata has none: page 1's largest-font lines."""
    first = [line for line in lines if line.page_number == 1 and line.horizontal
             and sum(ch.isalpha() for ch in line.text) >= 3 and not _ARXIV_STAMP_RE.match(line.text)]
    if not first:
        return None
    largest = max(line.size for line in first)
    if largest <= body_size(lines) + MIN_HEADING_SIZE_DELTA:
        return None
    title = " ".join(line.text for line in first if line.size >= largest - 0.5)
    return re.sub(r"(?<=[a-z])- (?=[a-z])", "", title)[:MAX_TITLE_CHARS] or None


def _is_heading_text(text: str) -> bool:
    """Short, mostly letters, starting with a capital or a number, and not a caption or arXiv stamp."""
    words = text.split()
    letters = sum(ch.isalpha() for ch in text)
    return (0 < len(words) <= MAX_HEADING_WORDS and letters >= 3 and letters >= 0.5 * len(text.replace(" ", ""))
            and (text[0].isupper() or text[0].isdigit())
            and not _CAPTION_RE.match(text) and not _ARXIV_STAMP_RE.match(text))


def font_headings(lines: list[_Line]) -> list[Heading]:
    """Lines larger than the body text (level by size rank), plus short standalone
    blocks that are bold or numbered (level by numbering depth). Excluded: running
    headers repeated across pages, rotated text, and page 1's largest (title) lines."""
    lines = [line for line in lines if line.horizontal]
    if not lines:
        return []
    body = body_size(lines)
    title_size = max((line.size for line in lines if line.page_number == 1), default=0.0)
    pages_per_text = Counter()
    for text, page in {(line.text, line.page_number) for line in lines}:
        pages_per_text[text] += 1
    large_sizes = sorted({round(line.size, 1) for line in lines
                          if body + MIN_HEADING_SIZE_DELTA <= line.size < title_size - 0.5}, reverse=True)

    headings: list[Heading] = []
    for line in lines:
        text = line.text
        if pages_per_text[text] >= RUNNING_HEADER_MIN_PAGES or not _is_heading_text(text):
            continue
        if line.page_number == 1 and line.size >= title_size - 0.5:
            continue
        standalone = line.block_lines <= 2 and line.block_words <= MAX_HEADING_WORDS
        if line.size >= body + MIN_HEADING_SIZE_DELTA:
            size = round(line.size, 1)
            level = min(large_sizes.index(size) + 1, 3) if size in large_sizes else 1
        elif standalone and (line.bold or text.lower().strip(" .:") in _NAMED_HEADINGS) and len(text.split()) <= 12:
            level = _numbering_depth(text)
        elif standalone and _NUMBERED_HEADING_RE.match(text) and "." not in text.rstrip(".")[8:]:
            level = _numbering_depth(text)
        else:
            continue
        headings.append(Heading(level, text, line.page_number))
    return headings


def _numbering_depth(text: str) -> int:
    """"3" → 1, "3.2" → 2, "3.2.1" → 3; letters ("A.") → 2; anything else → 1."""
    if m := re.match(r"^(\d+(?:\.\d+)*)", text):
        return min(m.group(1).count(".") + 1, 3)
    return 2 if re.match(r"^[A-H]\.\s", text) else 1


# ── Preprocessors ─────────────────────────────────────────────────────────────

class Preprocessor(ABC):
    name: str

    @abstractmethod
    def clean(self, text: str) -> str:
        """Normalize one page's raw text."""


class BasicPreprocessor(Preprocessor):
    """Unicode NFKC (expands ligatures like "ﬁ"), drops soft hyphens and control
    characters, rejoins words hyphenated across a line break ("estima-\\ntion"),
    unwraps lines within a paragraph, and collapses runs of spaces. Paragraph
    breaks (blank lines) are kept."""

    name = "basic"

    _HYPHEN_BREAK_RE = re.compile(r"(?<=[a-z])-\n(?=[a-z])")
    _CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f­]")
    _SPACES_RE = re.compile(r"[ \t ]+")

    def clean(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = self._CONTROL_RE.sub("", text)
        text = self._HYPHEN_BREAK_RE.sub("", text)
        paragraphs = [" ".join(line.strip() for line in p.split("\n") if line.strip()) for p in text.split("\n\n")]
        return "\n\n".join(self._SPACES_RE.sub(" ", p) for p in paragraphs if p)


PARSERS: dict[str, type[Parser]] = {"pymupdf": PyMuPdfParser}
PREPROCESSORS: dict[str, type[Preprocessor]] = {"basic": BasicPreprocessor}


def make_parser(name: str) -> Parser:
    if name not in PARSERS:
        raise ValueError(f"Unknown parser {name!r}; expected one of {', '.join(PARSERS)}")
    return PARSERS[name]()


def make_preprocessor(name: str) -> Preprocessor:
    if name not in PREPROCESSORS:
        raise ValueError(f"Unknown preprocessor {name!r}; expected one of {', '.join(PREPROCESSORS)}")
    return PREPROCESSORS[name]()


# ── Sections ──────────────────────────────────────────────────────────────────

def _heading_pattern(title: str) -> Optional[re.Pattern]:
    """The heading's words in order, separated by anything non-alphanumeric."""
    words = re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", title))
    return re.compile(r"[\W_]*".join(map(re.escape, words)), re.IGNORECASE) if words else None


def _line_start(content: str, pos: int) -> Optional[int]:
    """Start of pos's line when only a section number (e.g. "3.2.") precedes pos on it."""
    start = content.rfind("\n", 0, pos) + 1
    return start if _HEADING_PREFIX_RE.match(content[start:pos]) else None


def locate_headings(content: str, page_starts: list[int], headings: list[Heading]) -> list[tuple[Heading, int]]:
    """(heading, start offset) for every heading found in order: searched from the
    previous heading onward, on its own page or the next, preferring a match at the
    start of a line. Headings that can't be found are skipped."""
    located: list[tuple[Heading, int]] = []
    cursor = 0
    bounds = page_starts + [len(content)]
    for heading in headings:
        pattern = _heading_pattern(heading.title)
        page_index = min(max(heading.page_number - 1, 0), len(page_starts) - 1)
        lo, hi = max(cursor, bounds[page_index]), bounds[min(page_index + 2, len(page_starts))]
        if pattern is None or lo >= hi:
            continue
        matches = list(pattern.finditer(content, lo, hi))
        at_line_start = [(m, s) for m in matches if (s := _line_start(content, m.start())) is not None and s >= lo]
        if at_line_start:
            start = at_line_start[0][1]
        elif matches:
            start = matches[0].start()
            if number := _TRAILING_NUMBER_RE.search(content, max(lo, start - 12), start):
                start = number.start()
        else:
            continue
        if located and start <= located[-1][1]:
            continue
        located.append((heading, start))
        cursor = start + 1
    return located


def _page_number_at(page_starts: list[int], offset: int) -> int:
    return sum(1 for s in page_starts if s <= offset)


def sections_from_headings(content: str, page_starts: list[int], located: list[tuple[Heading, int]],
                           front_title: str) -> list[Section]:
    """Each located heading starts a section ending where the next begins; text before
    the first heading is the front-matter section."""
    starts = [(Heading(1, front_title, 1), 0)] if located[0][1] > 0 else []
    starts += located
    bounds = [s for _, s in starts] + [len(content)]
    return [Section(index=i, title=h.title, level=h.level, page_number=_page_number_at(page_starts, start),
                    start_char=start, end_char=bounds[i + 1])
            for i, (h, start) in enumerate(starts)]


def page_sections(content: str, page_starts: list[int]) -> list[Section]:
    bounds = page_starts + [len(content)]
    spans = [(i + 1, bounds[i], bounds[i + 1]) for i in range(len(page_starts)) if bounds[i + 1] > bounds[i]]
    return [Section(index=i, title=f"Page {page}", level=1, page_number=page, start_char=s, end_char=e)
            for i, (page, s, e) in enumerate(spans)]


def find_sections(content: str, page_starts: list[int], raw: RawDocument) -> tuple[str, list[Section]]:
    """(section_source, sections) from the first source locating MIN_LOCATED_HEADINGS headings."""
    for source, headings in (("toc", raw.toc_headings), ("font", raw.font_headings)):
        located = locate_headings(content, page_starts, headings)
        if len(located) >= MIN_LOCATED_HEADINGS:
            return source, sections_from_headings(content, page_starts, located, FRONT_MATTER_TITLE)
    return "page", page_sections(content, page_starts)


# ── Documents ─────────────────────────────────────────────────────────────────

def build_document(raw: RawDocument, preprocessor: Preprocessor, has_qrels: bool = False) -> Document:
    """Clean every page, join them with PAGE_SEPARATOR, and locate sections."""
    pages = [preprocessor.clean(p) for p in raw.pages]
    if not any(p.strip() for p in pages):
        raise EmptyDocumentError(f"{raw.source} has no extractable text (scanned or image-only PDF?)")
    page_starts, offset = [], 0
    for page in pages:
        page_starts.append(offset)
        offset += len(page) + len(PAGE_SEPARATOR)
    content = PAGE_SEPARATOR.join(pages)
    section_source, sections = find_sections(content, page_starts, raw)
    return Document(
        id=Document.make_id(raw.source),
        content=content,
        metadata=DocumentMetadata(
            source=raw.source, arxiv_id=raw.arxiv_id, title=raw.title, author=raw.author, page_count=len(pages),
            parser=raw.parser, has_qrels=has_qrels, page_starts=page_starts, section_source=section_source,
            sections=sections,
        ),
    )


def parse_papers(paths: list[Path], parser: Parser, preprocessor: Preprocessor, qrel_ids: set[str],
                 log: Optional[Logger] = None) -> list[Document]:
    """Parse every PDF; failures are logged and skipped so one bad file never stops the run."""
    documents = []
    for path in paths:
        try:
            documents.append(build_document(parser.parse(path), preprocessor, path.stem in qrel_ids))
        except EmptyDocumentError as e:
            if log:
                log.event("document_empty", source=path.name, error=str(e))
        except Exception as e:
            if log:
                log.event("parse_failed", source=path.name, error=f"{type(e).__name__}: {e}")
    return documents
