"""
Query phase step 1: select the test queries from the benchmark.

QUERY_COUNT queries are drawn (with QUERY_SEED) from those whose qrels paper was
ingested — a query about a paper that isn't indexed can't be answered — with at
most one query per paper. QUERY_ABSTRACTIVE are abstractive and the rest
extractive; sources follow QUERY_SOURCE_MIX, rounded by largest remainder
(12 at 65/25/5/5% → 8 text, 3 text-image, 1 text-table, 0 text-table-image).
Each source's queries are split between the two types in proportion, scarce
sources first; the most plentiful source takes whatever the type totals still need.

Each selected query carries its ground truth: the qrels doc_id and section_id and
the reference answer from answers.json.

Output:
    data/queries/query-set_{iteration}.json
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path

from models import QueryRecord, QuerySet

TYPES = ("abstractive", "extractive")


def parse_mix(value: str) -> dict[str, float]:
    """"text:0.65,text-image:0.25" → {"text": 0.65, "text-image": 0.25}; weights must sum to 1."""
    mix = {}
    for part in value.split(","):
        if part.strip():
            name, _, weight = part.partition(":")
            mix[name.strip()] = float(weight)
    if not mix or abs(sum(mix.values()) - 1.0) > 1e-6:
        raise ValueError(f"QUERY_SOURCE_MIX weights must sum to 1, got {value!r}")
    return mix


def largest_remainder(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Integer counts summing to `total`, proportional to `weights`: floors first, then
    one more each to the largest fractional remainders (ties in the order given)."""
    exact = {k: total * w for k, w in weights.items()}
    counts = {k: math.floor(v) for k, v in exact.items()}
    by_remainder = sorted(exact, key=lambda k: -(exact[k] - counts[k]))
    for k in by_remainder[:total - sum(counts.values())]:
        counts[k] += 1
    return counts


def cell_quotas(source_counts: dict[str, int], abstractive: int,
                available: dict[tuple[str, str], int]) -> dict[tuple[str, str], int]:
    """(type, source) → how many to pick. Sources are visited from scarcest to most
    plentiful; each takes its proportional share of abstractive queries (within what's
    available and what's left), and the last source takes the remaining type totals."""
    total = sum(source_counts.values())
    left = {"abstractive": abstractive, "extractive": total - abstractive}
    sources = sorted((s for s, n in source_counts.items() if n), key=lambda s: sum(available.get((t, s), 0) for t in TYPES))
    quotas: dict[tuple[str, str], int] = {}
    for i, source in enumerate(sources):
        n = source_counts[source]
        avail_abs, avail_ext = available.get(("abstractive", source), 0), available.get(("extractive", source), 0)
        if i == len(sources) - 1:
            n_abs = left["abstractive"]
        else:
            n_abs = min(round(n * abstractive / total), left["abstractive"], avail_abs, n)
            n_abs = max(n_abs, n - min(avail_ext, left["extractive"]))     # too few extractive: use abstractive
        n_ext = n - n_abs
        if n_abs > avail_abs or n_ext > avail_ext or n_abs > left["abstractive"] or n_ext > left["extractive"]:
            raise ValueError(f"Not enough {source} queries for {n_abs} abstractive + {n_ext} extractive "
                             f"(available: {avail_abs} + {avail_ext})")
        quotas[("abstractive", source)], quotas[("extractive", source)] = n_abs, n_ext
        left["abstractive"] -= n_abs
        left["extractive"] -= n_ext
    return quotas


def candidate_queries(queries: dict, qrels: dict, answers: dict, doc_ids: set[str]) -> list[QueryRecord]:
    """Every benchmark query whose qrels paper is in `doc_ids`, sorted by query id."""
    return [QueryRecord(query_id=qid, query=queries[qid]["query"], type=queries[qid]["type"],
                        source=queries[qid]["source"], doc_id=rel["doc_id"], section_id=rel["section_id"],
                        reference_answer=answers.get(qid, ""))
            for qid, rel in sorted(qrels.items()) if rel["doc_id"] in doc_ids and qid in queries]


def select_queries(candidates: list[QueryRecord], count: int, abstractive: int, mix: dict[str, float],
                   seed: int) -> list[QueryRecord]:
    """`count` queries meeting the type and source quotas, at most one per paper.
    Scarce cells pick first so plentiful ones can avoid their papers."""
    if not 0 <= abstractive <= count:
        raise ValueError(f"QUERY_ABSTRACTIVE must be between 0 and QUERY_COUNT ({count}), got {abstractive}")
    cells: dict[tuple[str, str], list[QueryRecord]] = defaultdict(list)
    for q in candidates:
        cells[(q.type, q.source)].append(q)
    quotas = cell_quotas(largest_remainder(count, mix), abstractive, {k: len(v) for k, v in cells.items()})

    rng = random.Random(seed)
    used_papers: set[str] = set()
    selected: list[QueryRecord] = []
    for cell in sorted((c for c, n in quotas.items() if n), key=lambda c: len(cells[c])):
        pool = cells[cell][:]
        rng.shuffle(pool)
        picked = []
        for q in pool:
            if len(picked) == quotas[cell]:
                break
            if q.doc_id not in used_papers:
                picked.append(q)
                used_papers.add(q.doc_id)
        if len(picked) < quotas[cell]:
            raise ValueError(f"Only {len(picked)} {cell[0]} {cell[1]} queries on unused papers; need {quotas[cell]}")
        selected += picked
    return selected


def build_query_set(queries_path: Path, qrels_path: Path, answers_path: Path, doc_ids: set[str], count: int,
                    abstractive: int, mix: dict[str, float], seed: int) -> QuerySet:
    with open(queries_path, encoding="utf-8") as f:
        queries = json.load(f)
    with open(qrels_path, encoding="utf-8") as f:
        qrels = json.load(f)
    with open(answers_path, encoding="utf-8") as f:
        answers = json.load(f)
    selected = select_queries(candidate_queries(queries, qrels, answers, doc_ids), count, abstractive, mix, seed)
    return QuerySet(seed=seed, count=count, abstractive=abstractive, source_mix=mix, queries=selected)
