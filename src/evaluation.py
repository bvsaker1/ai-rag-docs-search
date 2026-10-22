"""
Evaluation phase components: ground truth, retrieval metrics, the LLM judge, and the
experiment tracker.

Ground truth. qrels.json maps each query to (doc_id, section_id); the section's text
comes from the corpus JSON (the reference "master sheet", never used for ingestion).
A chunk of that paper is relevant when at least RELEVANCE_MIN_OVERLAP of its word
RELEVANCE_NGRAM-grams occur in the section, or of the section's occur in the chunk —
both directions, because some sections are a few dozen words that a 512-token chunk
contains whole.

Metrics at K (binary relevance), section level:
    Recall@K     relevant chunks in the top K / all relevant chunks
    Precision@K  relevant chunks in the top K / K   (max_precision = min(K, relevant) / K)
    MRR          1 / rank of the first relevant chunk (0 if none in the top K)
    NDCG@K       DCG of the top K / DCG of the ideal top K
    Hit@K        1 if any relevant chunk is in the top K
Paper level (secondary): every chunk of the qrels paper counts as relevant
(Precision@K, MRR, NDCG@K, Hit@K).
"""

import ast
import functools
import hashlib
import json
import math
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from api_retry import call_with_retries
from config import JUDGE_TEMPERATURE, LLM_MAX_ATTEMPTS, RELEVANCE_MIN_OVERLAP, RELEVANCE_NGRAM
from generation import VALIDATION_RETRIES, load_template
from models import AnswerFile, Chunk, EvaluationResult, JudgeScores, Judgment, QAResponse, QueryRecord
from saving import load_model, read_jsonl, write_jsonl

# Each source shown to the judge is cut to this many characters.
JUDGE_SOURCE_CHARS = 2000
_WORD_RE = re.compile(r"[a-z0-9]+")


# ── Ground truth ──────────────────────────────────────────────────────────────

@functools.cache
def corpus_sections(corpus_dir: Path, doc_id: str) -> dict[int, str]:
    """section_id → text for one paper of the corpus."""
    with open(corpus_dir / f"{doc_id}.json", encoding="utf-8") as f:
        sections = json.load(f)["sections"]
    if isinstance(sections, str):
        sections = ast.literal_eval(sections)
    return {int(s["section_id"]): s["text"] for s in sections}


def ngrams(text: str, n: int = RELEVANCE_NGRAM) -> set[tuple[str, ...]]:
    words = _WORD_RE.findall(text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def overlap(chunk_grams: set, section_grams: set) -> float:
    """The larger of: share of the chunk found in the section, share of the section found in the chunk."""
    if not chunk_grams or not section_grams:
        return 0.0
    shared = len(chunk_grams & section_grams)
    return max(shared / len(chunk_grams), shared / len(section_grams))


def relevant_chunk_ids(query: QueryRecord, chunks: list[Chunk], corpus_dir: Path,
                       min_overlap: float = RELEVANCE_MIN_OVERLAP, n: int = RELEVANCE_NGRAM) -> set[str]:
    """Ids of the chunks of the query's paper that overlap its qrels section."""
    section = ngrams(corpus_sections(corpus_dir, query.doc_id).get(query.section_id, ""), n)
    return {str(c.id) for c in chunks
            if c.metadata.arxiv_id == query.doc_id and overlap(ngrams(c.content, n), section) >= min_overlap}


# ── Metrics ───────────────────────────────────────────────────────────────────

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / k


def reciprocal_rank(retrieved: list[str], relevant: set[str], k: int) -> float:
    return next((1.0 / i for i, cid in enumerate(retrieved[:k], 1) if cid in relevant), 0.0)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, cid in enumerate(retrieved[:k], 1) if cid in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal else 0.0


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


class MetricsCalculator(ABC):
    @abstractmethod
    def score(self, retrieved: list[str], relevant: set[str], k: int) -> dict[str, float]:
        """Metrics for one query: `retrieved` chunk ids best first, `relevant` the ground truth."""


class IRMetricsCalculator(MetricsCalculator):
    def score(self, retrieved: list[str], relevant: set[str], k: int) -> dict[str, float]:
        return {
            "recall": recall_at_k(retrieved, relevant, k),
            "precision": precision_at_k(retrieved, relevant, k),
            "mrr": reciprocal_rank(retrieved, relevant, k),
            "ndcg": ndcg_at_k(retrieved, relevant, k),
            "hit": hit_at_k(retrieved, relevant, k),
        }


# ── LLM judge ─────────────────────────────────────────────────────────────────

class AnswerJudge(ABC):
    name: str

    @abstractmethod
    def judge(self, response: QAResponse, reference_answer: str) -> JudgeScores:
        """Relevance, accuracy, completeness, and citation-quality scores."""


class LLMAnswerJudge(AnswerJudge):
    """Scores an answer against the reference answer and the sources it was given
    (templates/judge_*.txt), through Instructor. `client` defaults to config.make_client()."""

    def __init__(self, model: str, client=None, temperature: float = JUDGE_TEMPERATURE,
                 max_attempts: int = LLM_MAX_ATTEMPTS, sleep: Callable[[float], None] = time.sleep) -> None:
        if client is None:
            from config import make_client
            client = make_client()
        self.name, self.model, self.client, self.temperature = model, model, client, temperature
        self.max_attempts, self._sleep = max_attempts, sleep
        self.system_prompt = load_template("judge_system.txt")
        self.user_template = load_template("judge_user.txt")

    def judge(self, response: QAResponse, reference_answer: str) -> JudgeScores:
        sources = "\n\n".join(f"[{i}] ({c.metadata.source}, page {c.metadata.page_number})\n"
                              f"{c.content[:JUDGE_SOURCE_CHARS]}" for i, c in enumerate(response.chunks_used, 1))
        user = self.user_template.format(query=response.query, reference_answer=reference_answer,
                                         answer=response.answer, sources=sources or "(none)")
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user}]

        def call() -> JudgeScores:
            return self.client.chat.completions.create(model=self.model, messages=messages, response_model=JudgeScores,
                                                       max_retries=VALIDATION_RETRIES, temperature=self.temperature)

        return call_with_retries(call, self.max_attempts, sleep=self._sleep)


def answer_hash(response: QAResponse) -> str:
    return hashlib.sha256(response.model_dump_json().encode()).hexdigest()[:16]


def judge_answers(files: list[Path], queries: dict[str, QueryRecord], judge: AnswerJudge, cache_path: Path,
                  concurrency: int) -> list[Judgment]:
    """A Judgment for every answered entry in `files`, reusing cached ones for the same answer;
    new ones are added to the cache (which keeps judgments of older answers too)."""
    cached = {j.key: j for j in (read_jsonl(cache_path, Judgment) if cache_path.exists() else []) if j.scores}
    todo, results = [], []
    for path in files:
        answers = load_model(path, AnswerFile)
        for entry in answers.entries:
            if entry.response is None:
                continue
            key = (answers.generator, answers.config.name, entry.query_id, judge.name, answer_hash(entry.response))
            if key in cached:
                results.append(cached[key])
            else:
                todo.append((key, entry.response))

    def run(item) -> Judgment:
        (gen, config, query_id, judge_name, digest), response = item
        judgment = Judgment(generator=gen, configuration=config, query_id=query_id, judge=judge_name,
                            answer_hash=digest)
        try:
            judgment.scores = judge.judge(response, queries[query_id].reference_answer)
        except Exception as e:
            judgment.error = f"{type(e).__name__}: {e}"
        return judgment

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        new = list(pool.map(run, todo))
    write_jsonl(cache_path, [*cached.values(), *(j for j in new if j.scores)])
    return results + new


# ── Experiment tracking ───────────────────────────────────────────────────────

class ExperimentTracker(ABC):
    @abstractmethod
    def record(self, result: EvaluationResult) -> None:
        """Keep one configuration's results."""

    @abstractmethod
    def results(self) -> list[EvaluationResult]:
        """Everything recorded so far."""


class JsonlExperimentTracker(ExperimentTracker):
    """One EvaluationResult per line in `path` (data/metrics_{iteration}.jsonl); a configuration
    recorded again replaces its earlier line."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._results = {r.experiment_id: r for r in (read_jsonl(path, EvaluationResult) if path.exists() else [])}

    def record(self, result: EvaluationResult) -> None:
        self._results[result.experiment_id] = result
        write_jsonl(self.path, self._results.values())

    def results(self) -> list[EvaluationResult]:
        return list(self._results.values())
