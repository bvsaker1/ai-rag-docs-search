from collections import Counter

import pytest

from models import QueryRecord
from query_set import cell_quotas, largest_remainder, parse_mix, select_queries

MIX = {"text": 0.65, "text-image": 0.25, "text-table": 0.05, "text-table-image": 0.05}


def test_parse_mix_requires_weights_summing_to_one():
    assert parse_mix("text:0.65, text-image:0.35") == {"text": 0.65, "text-image": 0.35}
    with pytest.raises(ValueError, match="sum to 1"):
        parse_mix("text:0.65,text-image:0.25,text-table:0.05")


def test_largest_remainder_rounds_twelve_queries():
    assert largest_remainder(12, MIX) == {"text": 8, "text-image": 3, "text-table": 1, "text-table-image": 0}
    assert sum(largest_remainder(7, MIX).values()) == 7


def test_cell_quotas_split_types_proportionally_and_respect_availability():
    available = {("abstractive", "text"): 100, ("extractive", "text"): 100, ("abstractive", "text-image"): 50,
                 ("extractive", "text-image"): 10, ("abstractive", "text-table"): 5}
    quotas = cell_quotas({"text": 8, "text-image": 3, "text-table": 1}, 8, available)
    assert quotas[("abstractive", "text-table")] == 1 and quotas[("extractive", "text-table")] == 0
    assert (quotas[("abstractive", "text-image")], quotas[("extractive", "text-image")]) == (2, 1)
    assert (quotas[("abstractive", "text")], quotas[("extractive", "text")]) == (5, 3)


def test_cell_quotas_fail_when_a_source_runs_out():
    with pytest.raises(ValueError, match="Not enough text-table"):
        cell_quotas({"text": 1, "text-table": 2}, 2, {("abstractive", "text"): 5, ("abstractive", "text-table"): 1})


def candidates(n_papers: int = 20) -> list[QueryRecord]:
    """Each paper gets one query of every type × source."""
    out = []
    for p in range(n_papers):
        for t in ("abstractive", "extractive"):
            for s in MIX:
                out.append(QueryRecord(query_id=f"{p}-{t}-{s}", query=f"q {p} {t} {s}", type=t, source=s,
                                       doc_id=f"paper{p}", section_id=0, reference_answer="a"))
    return out


def test_select_queries_meets_quotas_one_per_paper_and_is_reproducible():
    picked = select_queries(candidates(), 12, 8, MIX, seed=42)
    assert len(picked) == 12 and len({q.doc_id for q in picked}) == 12
    assert Counter(q.type for q in picked) == {"abstractive": 8, "extractive": 4}
    assert Counter(q.source for q in picked) == {"text": 8, "text-image": 3, "text-table": 1}
    assert [q.query_id for q in select_queries(candidates(), 12, 8, MIX, seed=42)] == [q.query_id for q in picked]
    assert [q.query_id for q in select_queries(candidates(), 12, 8, MIX, seed=7)] != [q.query_id for q in picked]


def test_select_queries_fails_without_enough_distinct_papers():
    with pytest.raises(ValueError, match="unused papers"):
        select_queries(candidates(n_papers=5), 12, 8, MIX, seed=1)
    with pytest.raises(ValueError, match="QUERY_ABSTRACTIVE"):
        select_queries(candidates(), 12, 13, MIX, seed=1)
