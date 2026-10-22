import numpy as np
import pytest

from chunking import (
    FixedTokenChunker, RecursiveChunker, SemanticChunker, Unit, build_config, build_units, chunk_documents,
    count_tokens, make_chunker, merged_section_spans, pack_groups, recursive_units, semantic_groups, sentence_spans,
    token_spans,
)
from conftest import FakeEmbedder, make_document
from parser import Heading

ENCODING = "cl100k_base"


def config(method: str, chunk_size: int = 10, overlap: int = 3, **kw):
    return build_config(method, chunk_size, overlap, parser="pymupdf", tokenizer=ENCODING, **kw)


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def sentences(n: int, topic: str = "grid") -> str:
    return " ".join(f"Sentence {i} is about the {topic}." for i in range(n))


# ── Tokens & sentences ────────────────────────────────────────────────────────

def test_token_spans_cover_the_text_in_order():
    text = "Impedance estimation in power grids."
    spans = token_spans(text, ENCODING)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


def test_sentence_spans_respect_abbreviations_and_paragraphs():
    text = "As shown in Fig. 3 the error drops. Smith et al. Agree.\n\nNew paragraph without a period"
    assert [text[s:e] for s, e in sentence_spans(text)] == [
        "As shown in Fig. 3 the error drops.", "Smith et al. Agree.", "New paragraph without a period"]


def test_units_never_cross_pages_and_long_sentences_are_split():
    doc = make_document(["Short one. " + words(25, "x") + ".", "Page two sentence."])
    units = build_units(doc, max_tokens=10, encoding=ENCODING)
    page_two = doc.metadata.page_starts[1]
    assert all(u.end <= page_two or u.start >= page_two for u in units)
    assert all(u.tokens <= 10 for u in units)
    assert doc.content[units[-1].start:units[-1].end] == "Page two sentence."


# ── Grouping ──────────────────────────────────────────────────────────────────

def units_of(*tokens: int) -> list[Unit]:
    return [Unit(i * 10, i * 10 + 5, t) for i, t in enumerate(tokens)]


def test_pack_groups_carry_trailing_units_that_fit_the_overlap_budget():
    units = units_of(3, 3, 3, 3, 3, 3, 3)
    assert pack_groups(units, max_tokens=9, overlap_tokens=6) == [(0, 3), (1, 4), (2, 5), (3, 6), (4, 7)]
    assert pack_groups(units, max_tokens=9, overlap_tokens=2) == [(0, 3), (3, 6), (6, 7)]   # nothing fits


def test_pack_groups_drop_overlap_that_would_leave_no_room():
    units = units_of(5, 5, 9)
    assert pack_groups(units, max_tokens=10, overlap_tokens=5) == [(0, 2), (2, 3)]


def test_semantic_groups_cut_at_high_distance_and_respect_limits():
    units = units_of(5, 5, 5, 5, 5, 5)
    distances = np.array([0.1, 0.9, 0.1, 0.1, 0.1])
    assert semantic_groups(units, distances, percentile=80, min_tokens=5, max_tokens=100) == [(0, 2), (2, 6)]
    assert semantic_groups(units, distances, percentile=80, min_tokens=5, max_tokens=10) == [(0, 2), (2, 4), (4, 6)]
    assert semantic_groups(units, distances, percentile=80, min_tokens=15, max_tokens=100) == [(0, 6)]


# ── Chunkers ──────────────────────────────────────────────────────────────────

def test_fixed_windows_have_chunk_size_tokens_and_overlap():
    doc = make_document([words(40)])
    chunks = FixedTokenChunker(config("FIXED", 10, 3)).chunk(doc)
    tokens = token_spans(doc.content, ENCODING)
    assert all(c.metadata.token_count <= 10 for c in chunks)
    assert chunks[-1].metadata.end_char == len(doc.content)
    for a, b in zip(chunks, chunks[1:]):
        shared = [t for t in tokens if t[0] < a.metadata.end_char and t[1] > b.metadata.start_char]
        assert len(shared) == 3


def test_every_chunk_is_its_documents_exact_slice():
    doc = make_document([sentences(30), sentences(30, "data")])
    for method in ("FIXED", "RECURSIVE", "SEMANTIC"):
        chunker = make_chunker(config(method, 40, 8), FakeEmbedder())
        for c in chunker.chunk(doc):
            assert doc.content[c.metadata.start_char:c.metadata.end_char] == c.content
            assert c.content == c.content.strip()


def test_recursive_units_split_only_as_far_as_needed():
    text = "First paragraph is short.\n\n" + sentences(6) + "\n\n" + words(40, "x")
    units = recursive_units(text, 0, len(text), max_tokens=30, encoding=ENCODING)
    pieces = [text[u.start:u.end] for u in units]
    assert pieces[0] == "First paragraph is short."                     # fits: kept whole
    assert pieces[1] == "Sentence 0 is about the grid."                 # paragraph too long: sentences
    assert all(u.tokens <= 30 for u in units)
    assert all(p.startswith("x") for p in pieces[7:])                   # one long sentence: words


def test_recursive_units_fall_back_to_token_windows():
    text = "x" * 400                                                     # no separators at all
    units = recursive_units(text, 0, len(text), max_tokens=20, encoding=ENCODING)
    assert len(units) > 1 and all(u.tokens <= 20 for u in units)
    assert units[0].start == 0 and units[-1].end == len(text)


def test_merged_section_spans_fold_tiny_sections_into_the_next():
    doc = make_document(["Intro\n\n" + sentences(8) + "\n\n3. Methods\n\n3.1 Setup\n\n" + sentences(8, "data")],
                        toc=[Heading(1, "Intro", 1), Heading(1, "Methods", 1), Heading(2, "Setup", 1)])
    assert [s.title for s in doc.metadata.sections] == ["Intro", "Methods", "Setup"]
    spans = merged_section_spans(doc, min_tokens=20)
    assert [doc.content[s:s + 10] for s, _ in spans] == ["Intro\n\nSen", "3. Methods"]
    assert spans[-1][1] == len(doc.content)


def test_recursive_chunks_stay_within_sections_and_overlap():
    doc = make_document(["Intro\n\n" + sentences(20), "Methods\n\n" + sentences(20, "data")],
                        toc=[Heading(1, "Intro", 1), Heading(1, "Methods", 2)])
    chunks = RecursiveChunker(config("RECURSIVE", 40, 10)).chunk(doc)
    boundary = doc.metadata.sections[1].start_char
    assert all(c.metadata.end_char <= boundary or c.metadata.start_char >= boundary for c in chunks)
    assert all(c.metadata.token_count <= 40 for c in chunks)
    same_section = [(a, b) for a, b in zip(chunks, chunks[1:]) if a.metadata.section_index == b.metadata.section_index]
    assert same_section and all(b.metadata.start_char < a.metadata.end_char for a, b in same_section)
    assert all(count_tokens(a.content[b.metadata.start_char - a.metadata.start_char:], ENCODING) <= 10
               for a, b in same_section)
    assert all(c.content.endswith(".") or c.content in ("Intro", "Methods") for c in chunks)


def test_semantic_chunks_split_where_the_topic_changes():
    doc = make_document([sentences(6, "grid impedance") + " " + sentences(6, "school enrollment")])
    chunks = SemanticChunker(config("SEMANTIC", 500, breakpoint_percentile=95, min_tokens=5), FakeEmbedder()).chunk(doc)
    assert len(chunks) == 2
    assert "enrollment" not in chunks[0].content and "impedance" not in chunks[1].content


def test_chunk_metadata_records_pages_and_sections():
    doc = make_document(["Intro\n\n" + sentences(8), "Methods\n\n" + sentences(8, "data")],
                        source="2401.00001v1.pdf", toc=[Heading(1, "Intro", 1), Heading(1, "Methods", 2)])
    chunks = FixedTokenChunker(config("FIXED", 60, 10)).chunk(doc)
    assert [c.metadata.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].metadata.section_title == "Intro" and chunks[-1].metadata.section_title == "Methods"
    assert any(c.metadata.page_number == 1 and c.metadata.end_page_number == 2 for c in chunks)
    first = chunks[0].metadata
    assert (first.document_id, first.source, first.arxiv_id) == (doc.id, "2401.00001v1.pdf", "2401.00001v1")


def test_chunk_documents_restarts_indexes_and_ids_are_stable():
    docs = [make_document([sentences(20)], source="a.pdf"), make_document([sentences(20)], source="b.pdf")]
    chunker = FixedTokenChunker(config("FIXED", 30, 5))
    chunks = chunk_documents(docs, chunker)
    assert [c.metadata.chunk_index for c in chunks if c.metadata.source == "b.pdf"][0] == 0
    assert len({c.id for c in chunks}) == len(chunks)
    assert [c.id for c in chunk_documents(docs, chunker)] == [c.id for c in chunks]


def test_build_config_and_make_chunker():
    assert config("recursive").separators == ["section", "paragraph", "sentence", "word", "token"]
    assert config("RECURSIVE").min_tokens == 50 and config("RECURSIVE").overlap == 3
    assert config("SEMANTIC").overlap == 0
    assert isinstance(make_chunker(config("FIXED")), FixedTokenChunker)
    with pytest.raises(ValueError, match="Unknown chunking method"):
        config("SENTENCE")
    with pytest.raises(ValueError, match="boundary embedder"):
        make_chunker(config("SEMANTIC"))
