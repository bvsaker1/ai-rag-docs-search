#!/usr/bin/env python3
"""
Compare generator LLMs on the query set before choosing GENERATION_MODEL.

For each model in GENERATION_CANDIDATES:
    1. a one-call smoke test (an Instructor answer for the first query) — a model
       that fails it is reported and skipped
    2. the full query phase: every configuration × every query
       → data/answers/{iteration}/{model}/answer-{config}.json
Then JUDGE_MODEL scores every answer on relevance, accuracy, completeness, and
citation quality (1-5). Judgments are cached by answer text, so reruns only judge
new or regenerated answers.

Retrieval is identical for every candidate, so score differences come from the generator.

Usage:
    python compare_generators.py
    python compare_generators.py --models anthropic/claude-haiku-5.5 openai/gpt-6-luna
    python compare_generators.py --force        # regenerate answers (judgments of old answers are reused)

Output:
    data/answers/{iteration}/judgments.jsonl
    data/generator_comparison_{iteration}.json
"""

import argparse
import functools
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from config import GENERATION_CANDIDATES, ITERATION, JUDGE_MODEL, Logger
from evaluation import AnswerJudge, judge_answers
from generation import Generator
from models import AnswerFile, Judgment, QueryRecord
from query_phase import QuerySettings, Resources, answer_entry, configurations, load_query_set, run_query_phase
from saving import generator_comparison_path, judgments_path, load_model, save_json

print = functools.partial(print, flush=True)

CRITERIA = ["relevance", "accuracy", "completeness", "citation_quality"]


def provider(model: str) -> str:
    return model.split("/")[0] if "/" in model else model


def smoke_test(generator: Generator, s: QuerySettings, query: QueryRecord) -> Optional[str]:
    """None when one answer comes back with a citation-ready response; else the error."""
    config = configurations(s)[0]
    retriever = Resources(s).retriever(config)
    entry = answer_entry(query, retriever.retrieve(query.query, config.top_k), 0.0, generator)
    return entry.error


def summarize(judgments: list[Judgment], files_by_generator: dict[str, list[Path]],
              queries: dict[str, QueryRecord]) -> dict:
    """Per generator: mean scores overall and by query type, answer errors, judge errors, latency, tokens."""
    summary = {}
    for generator, files in files_by_generator.items():
        entries = [e for p in files for e in load_model(p, AnswerFile).entries]
        scored = [j for j in judgments if j.generator == generator and j.scores]
        row = {
            "answers": len(entries),
            "answer_errors": sum(e.error is not None for e in entries),
            "judged": len(scored),
            "judge_errors": sum(1 for j in judgments if j.generator == generator and j.error),
            "mean_generation_seconds": round(statistics.mean(e.generation_seconds for e in entries), 2) if entries else None,
            "prompt_tokens": sum(e.prompt_tokens or 0 for e in entries),
            "completion_tokens": sum(e.completion_tokens or 0 for e in entries),
        }
        for criterion in CRITERIA:
            row[criterion] = round(statistics.mean(getattr(j.scores, criterion) for j in scored), 3) if scored else None
        row["overall"] = round(statistics.mean(j.scores.overall for j in scored), 3) if scored else None
        for qtype in ("abstractive", "extractive"):
            typed = [j for j in scored if queries[j.query_id].type == qtype]
            row[f"overall_{qtype}"] = round(statistics.mean(j.scores.overall for j in typed), 3) if typed else None
        summary[generator] = row
    return summary


def print_table(summary: dict) -> None:
    cols = ["overall", *CRITERIA, "overall_abstractive", "overall_extractive", "answer_errors",
            "mean_generation_seconds", "completion_tokens"]
    print("\n| generator | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for generator, row in sorted(summary.items(), key=lambda kv: -(kv[1]["overall"] or 0)):
        print(f"| {generator} | " + " | ".join(str(row[c]) for c in cols) + " |")


def compare(models: list[str], s: QuerySettings, make_generator: Callable[[str], Generator],
            judge: AnswerJudge) -> dict:
    log = Logger(s.iteration, "compare_generators", path=s.log_path, judge=judge.name)
    try:
        query_set = load_query_set(s, log)
        queries = {q.query_id: q for q in query_set.queries}
        log.event("step_start", candidates=models, queries=len(queries), configurations=len(configurations(s)))
        same_provider = [m for m in models if provider(m) == provider(judge.name)]
        if same_provider:
            print(f"WARNING: judge {judge.name} shares a provider with {same_provider}; its scores may be biased")

        files_by_generator: dict[str, list[Path]] = {}
        for model in models:
            generator = make_generator(model)
            error = smoke_test(generator, s, query_set.queries[0])
            log.event("smoke_test", generator=model, error=error)
            if error:
                print(f"[{model}] smoke test failed, skipping: {error}")
                continue
            print(f"[{model}] smoke test passed")
            files_by_generator[model] = run_query_phase(s, generator)

        start = time.monotonic()
        all_files = [p for files in files_by_generator.values() for p in files]
        judgments = judge_answers(all_files, queries, judge, judgments_path(s.iteration, s.data_dir), s.concurrency)
        log.event("judging_complete", judgments=len(judgments), errors=sum(j.error is not None for j in judgments),
                  duration_seconds=round(time.monotonic() - start, 2))

        summary = summarize(judgments, files_by_generator, queries)
        out = generator_comparison_path(s.iteration, s.data_dir)
        save_json(out, {"iteration": s.iteration, "judge": judge.name, "queries": len(queries),
                        "configurations": [c.name for c in configurations(s)], "generators": summary})
        log.event("step_complete", output=str(out), summary=summary)
        print_table(summary)
        print(f"\nComparison → {out}")
        return summary
    finally:
        log.close()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare generator LLMs with an LLM judge.")
    parser.add_argument("--models", nargs="+", default=GENERATION_CANDIDATES,
                        help=f"Generators to compare (default: {' '.join(GENERATION_CANDIDATES)})")
    parser.add_argument("--judge", default=JUDGE_MODEL, help=f"Judge model (default: {JUDGE_MODEL})")
    parser.add_argument("--iteration", type=int, default=ITERATION)
    parser.add_argument("--force", action="store_true", help="Regenerate answer files that already exist")
    args = parser.parse_args(argv)

    from evaluation import LLMAnswerJudge
    from generation import LLMGenerator

    s = QuerySettings(iteration=args.iteration, force=args.force)
    summary = compare(args.models, s, LLMGenerator, LLMAnswerJudge(args.judge))
    return 0 if summary else 1


if __name__ == "__main__":
    sys.exit(main())
