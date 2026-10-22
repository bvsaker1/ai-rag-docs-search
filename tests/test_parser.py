import pytest

from conftest import BODY, BOLD, make_document, write_pdf
from config import Logger
from models import PAGE_SEPARATOR
from parser import (
    BasicPreprocessor, EmptyDocumentError, Heading, PyMuPdfParser, RawDocument, build_document, find_sections,
    locate_headings, make_parser, make_preprocessor, page_sections, parse_papers, qrels_doc_ids, select_papers,
    toc_headings,
)


# ── Paper selection ───────────────────────────────────────────────────────────

def test_select_papers_takes_qrels_papers_first_then_negatives(tmp_path):
    for name in ["2401.00003v1", "2401.00001v1", "2401.00002v1", "2401.00004v1"]:
        (tmp_path / f"{name}.pdf").write_bytes(b"")
    qrel_ids = {"2401.00002v1", "2401.00004v1", "9999.99999v1"}
    assert [p.stem for p in select_papers(tmp_path, qrel_ids, 5)] == ["2401.00002v1", "2401.00004v1"]
    assert [p.stem for p in select_papers(tmp_path, qrel_ids, 1)] == ["2401.00002v1"]
    assert [p.stem for p in select_papers(tmp_path, qrel_ids, 1, max_negatives=1)] == ["2401.00002v1", "2401.00001v1"]


def test_qrels_doc_ids(qrels_file):
    assert qrels_doc_ids(qrels_file) == {"2401.00001v1", "2401.00002v1"}


# ── Preprocessing ─────────────────────────────────────────────────────────────

def test_preprocessor_fixes_ligatures_hyphens_and_wrapped_lines():
    raw = "The ﬁrst estima-\ntion is\nreported   here.\x00\n\nNext para-\ngraph."
    assert BasicPreprocessor().clean(raw) == "The first estimation is reported here.\n\nNext paragraph."


def test_preprocessor_keeps_hyphens_that_are_not_line_breaks():
    assert BasicPreprocessor().clean("state-of-the-art and Fig-\n3") == "state-of-the-art and Fig- 3"


# ── Sections ──────────────────────────────────────────────────────────────────

def test_toc_headings_drop_blank_titles_and_unresolved_pages():
    toc = [[1, "Introduction", 1], [1, " ", 7], [2, "Lost", -1], [2, "Model  Setup", 3]]
    assert toc_headings(toc) == [Heading(1, "Introduction", 1), Heading(2, "Model Setup", 3)]


def test_locate_headings_prefers_line_starts_and_skips_missing():
    content = "Our methods are new.\n\n3. Methods\n\nWe use data.\n\nResults\n\nIt works."
    headings = [Heading(1, "Methods", 1), Heading(1, "Missing Section", 1), Heading(1, "Results", 1)]
    located = locate_headings(content, [0], headings)
    assert [(h.title, content[s:s + 10]) for h, s in located] == [("Methods", "3. Methods"), ("Results", "Results\n\nI")]


def test_locate_headings_searches_in_order_and_on_the_headings_page():
    pages = ["Intro text. Methods appear early.", "1 Intro\n\nMore.", "2 Methods\n\nDone."]
    doc = make_document(pages)
    located = locate_headings(doc.content, doc.metadata.page_starts, [Heading(1, "Intro", 2), Heading(1, "Methods", 3)])
    assert [doc.content[s:s + 9] for _, s in located] == ["1 Intro\n\n", "2 Methods"]


def test_sections_tile_the_document_with_front_matter_first():
    doc = make_document(["Title line\n\n1. Introduction\n\nText.", "2. Methods\n\nMore text."],
                        toc=[Heading(1, "Introduction", 1), Heading(1, "Methods", 2)])
    sections = doc.metadata.sections
    assert doc.metadata.section_source == "toc"
    assert [(s.title, s.page_number) for s in sections] == [("Front matter", 1), ("Introduction", 1), ("Methods", 2)]
    assert sections[0].start_char == 0 and sections[-1].end_char == len(doc.content)
    assert all(a.end_char == b.start_char for a, b in zip(sections, sections[1:]))


def test_sections_fall_back_from_toc_to_font_to_pages():
    pages = ["1. Introduction\n\nText.", "2. Methods\n\nText."]
    font = [Heading(1, "1. Introduction", 1), Heading(1, "2. Methods", 2)]
    unfindable_toc = [Heading(1, "Nowhere", 1), Heading(1, "Gone", 2)]
    raw = RawDocument(source="a.pdf", pages=pages, parser="pymupdf", toc_headings=unfindable_toc, font_headings=font)
    content = PAGE_SEPARATOR.join(pages)
    starts = [0, len(pages[0]) + len(PAGE_SEPARATOR)]
    assert find_sections(content, starts, raw)[0] == "font"
    raw.font_headings = font[:1]          # one heading isn't enough
    source, sections = find_sections(content, starts, raw)
    assert source == "page" and sections == page_sections(content, starts)
    assert [s.title for s in sections] == ["Page 1", "Page 2"]


def test_page_sections_skip_an_empty_last_page():
    content = "Only page." + PAGE_SEPARATOR
    assert [s.title for s in page_sections(content, [0, len(content)])] == ["Page 1"]


# ── Documents ─────────────────────────────────────────────────────────────────

def test_build_document_joins_pages_and_records_offsets():
    doc = make_document(["Page one text.", "Page two text."])
    assert doc.content == "Page one text.\n\nPage two text."
    assert doc.metadata.page_starts == [0, 16]
    assert doc.metadata.page_count == 2 and doc.metadata.arxiv_id == "0000.00001v1"


def test_build_document_rejects_documents_without_text():
    with pytest.raises(EmptyDocumentError):
        make_document(["", "  \n "])


# ── PyMuPDF ───────────────────────────────────────────────────────────────────

def test_pymupdf_parser_reads_toc_title_and_author(papers_dir):
    raw = PyMuPdfParser().parse(papers_dir / "2401.00001v1.pdf")
    assert raw.parser == "pymupdf" and len(raw.pages) == 2
    assert [h.title for h in raw.toc_headings] == ["Introduction", "Methods", "Results", "Conclusion"]
    assert raw.title == "A Study of Grid Estimation"          # no metadata title: page 1's largest text
    assert "1. Introduction" in raw.pages[0]

    raw = PyMuPdfParser().parse(papers_dir / "2401.00002v1.pdf")
    assert (raw.title, raw.author) == ("Enrollment Paper", "A. Author")


def test_pymupdf_font_headings_find_bold_numbered_lines_but_not_the_title(papers_dir):
    raw = PyMuPdfParser().parse(papers_dir / "2401.00002v1.pdf")
    assert raw.toc_headings == []
    assert [(h.title, h.page_number) for h in raw.font_headings] == [
        ("1. Introduction", 1), ("2. Methods", 1), ("3. Results", 2), ("4. Conclusion", 2)]
    doc = build_document(raw, BasicPreprocessor())
    assert doc.metadata.section_source == "font"
    assert doc.metadata.sections[0].title == "Front matter"


def test_font_headings_skip_running_headers(tmp_path):
    page = [("Journal of Tests", 13, BOLD), ("Plain body text goes here.", 10, BODY)]
    raw = PyMuPdfParser().parse(write_pdf(tmp_path / "x.pdf", [page, page, page]))
    assert raw.font_headings == []


def test_parse_papers_skips_and_logs_broken_and_empty_pdfs(papers_dir, tmp_path):
    (papers_dir / "2401.00009v1.pdf").write_bytes(b"not a pdf")
    write_pdf(papers_dir / "2401.00010v1.pdf", [[]])
    log_file = tmp_path / "log.jsonl"
    log = Logger(1, "parsing", path=log_file)
    paths = sorted(papers_dir.glob("*.pdf"))
    docs = parse_papers(paths, PyMuPdfParser(), BasicPreprocessor(), {"2401.00001v1"}, log)
    log.close()
    assert [d.metadata.arxiv_id for d in docs] == ["2401.00001v1", "2401.00002v1", "2401.00003v1"]
    assert [d.metadata.has_qrels for d in docs] == [True, False, False]
    events = log_file.read_text()
    assert '"parse_failed"' in events and "2401.00009v1.pdf" in events
    assert '"document_empty"' in events and "2401.00010v1.pdf" in events


def test_factories_reject_unknown_names():
    assert isinstance(make_parser("pymupdf"), PyMuPdfParser)
    assert isinstance(make_preprocessor("basic"), BasicPreprocessor)
    with pytest.raises(ValueError, match="pdfplumber"):
        make_parser("pdfplumber")
    with pytest.raises(ValueError):
        make_preprocessor("fancy")
