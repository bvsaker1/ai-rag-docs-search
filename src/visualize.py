"""
Evaluation charts (matplotlib), saved as PNG to data/visualizations/{iteration}/:

    retrieval_heatmap.png      Recall@K, Precision@K, MRR, NDCG@K for every configuration (YlGnBu, values in cells)
    dimension_impact.png       mean NDCG@K per chunking method, embedding model, and retrieval method
                               (embedding means use dense and hybrid only — BM25 has no embeddings)
    generation_radar.png       mean judge scores of the best configuration
    query_latency.png          retrieval and end-to-end latency per configuration (box plots)
    before_after.png           the best configuration's metrics before vs after an improvement, with deltas
"""

import textwrap
from pathlib import Path
from statistics import mean
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from config import model_key  # noqa: E402
from models import EvaluationResult  # noqa: E402

METRICS = ["recall", "precision", "mrr", "ndcg"]
CRITERIA = ["relevance", "accuracy", "completeness", "citation_quality"]
DPI = 150


def metric_labels(k: int) -> list[str]:
    return [f"Recall@{k}", f"Precision@{k}", "MRR", f"NDCG@{k}"]


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def retrieval_heatmap(results: list[EvaluationResult], path: Path) -> Path:
    results = sorted(results, key=lambda r: -r.metrics["ndcg"])
    values = np.array([[r.metrics[m] for m in METRICS] for r in results])
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(results) + 1.8))
    image = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(METRICS)), metric_labels(results[0].k), fontsize=11)
    ax.set_yticks(range(len(results)), [r.experiment_id for r in results], fontsize=10)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if values[i, j] > 0.6 else "black")
    ax.set_xlabel("Metric", fontsize=11)
    ax.set_ylabel("Configuration (sorted by NDCG)", fontsize=11)
    ax.set_title(f"Retrieval metrics by configuration — iteration {results[0].iteration}", fontsize=13)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02, label="Score")
    return _save(fig, path)


def dimension_means(results: list[EvaluationResult], metric: str = "ndcg") -> dict[str, dict[str, float]]:
    """Mean `metric` per level of each dimension, averaged over the other dimensions."""
    def by(key, rows):
        groups: dict[str, list[float]] = {}
        for r in rows:
            groups.setdefault(key(r), []).append(r.metrics[metric])
        return {k: mean(v) for k, v in groups.items()}

    embedded = [r for r in results if r.config.embedding_model]
    return {
        "Chunking": by(lambda r: r.config.chunk_method, results),
        "Embedding (dense + hybrid)": by(lambda r: model_key(r.config.embedding_model), embedded),
        "Retrieval": by(lambda r: r.config.retrieval_method.value, results),
    }


def dimension_impact(results: list[EvaluationResult], path: Path) -> Path:
    means = dimension_means(results)
    fig, ax = plt.subplots(figsize=(10, 5))
    x, ticks, labels, colors = 0.0, [], [], plt.get_cmap("tab10").colors
    for d, (dimension, levels) in enumerate(means.items()):
        start = x
        for level, value in levels.items():
            ax.bar(x, value, width=0.8, color=colors[d])
            ax.text(x, value + 0.01, f"{value:.2f}", ha="center", fontsize=10)
            ticks.append(x)
            labels.append(level)
            x += 1
        spread = max(levels.values()) - min(levels.values())
        ax.text((start + x - 1) / 2, -0.17, f"{dimension}\n(spread {spread:.2f})", ha="center", va="top",
                fontsize=11, transform=ax.get_xaxis_transform())
        x += 1
    k = results[0].k
    ax.set_xticks(ticks, labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"Mean NDCG@{k}", fontsize=11)
    ax.set_title(f"Configuration dimension impact on NDCG@{k} — iteration {results[0].iteration}", fontsize=13)
    return _save(fig, path)


def generation_radar(best: EvaluationResult, path: Path) -> Path:
    values = [best.judge_scores[c] for c in CRITERIA]
    angles = np.linspace(0, 2 * np.pi, len(CRITERIA), endpoint=False).tolist()
    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw={"polar": True})
    ax.plot(angles + angles[:1], values + values[:1], linewidth=2, color="tab:blue")
    ax.fill(angles + angles[:1], values + values[:1], alpha=0.25, color="tab:blue")
    ax.set_xticks(angles, ["Relevance", "Accuracy", "Completeness", "Citation\nquality"], fontsize=11)
    ax.tick_params(axis="x", pad=18)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], fontsize=9)
    for angle, value in zip(angles, values):
        ax.text(angle, value - 0.7, f"{value:.2f}", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.set_title(f"Generation quality (LLM judge, 1-5) — best configuration\n{best.experiment_id}, "
                 f"{best.generator}", fontsize=12, pad=20)
    return _save(fig, path)


def query_latency(results: list[EvaluationResult], path: Path) -> Path:
    results = sorted(results, key=lambda r: r.experiment_id)
    names = [r.experiment_id for r in results]
    retrieval = [[q.retrieval_ms for q in r.per_query if q.retrieval_ms is not None] for r in results]
    total = [[q.retrieval_ms / 1000 + q.generation_seconds for q in r.per_query
              if q.retrieval_ms is not None and q.generation_seconds is not None] for r in results]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    top.boxplot(retrieval, showfliers=True)
    top.set_yscale("log")
    top.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    top.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    top.set_ylabel("Retrieval latency (ms, log scale)", fontsize=11)
    top.set_title("Query latency by configuration (warm retrieval; end-to-end adds LLM generation)", fontsize=13)
    bottom.boxplot(total, showfliers=True)
    bottom.axhline(5, color="red", linestyle="--", linewidth=1)
    bottom.text(0.6, 5.1, "5 s target", color="red", fontsize=10)
    bottom.set_ylabel("End-to-end latency (s)", fontsize=11)
    bottom.set_xticks(range(1, len(names) + 1), names, rotation=45, ha="right", fontsize=10)
    bottom.set_xlabel("Configuration", fontsize=11)
    return _save(fig, path)


def before_after(before: EvaluationResult, after: EvaluationResult, path: Path,
                 change: Optional[str] = None) -> Path:
    labels = metric_labels(after.k)
    x = np.arange(len(METRICS))
    b = [before.metrics[m] for m in METRICS]
    a = [after.metrics[m] for m in METRICS]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - 0.2, b, width=0.4, label=f"Before: iteration {before.iteration} ({before.experiment_id})",
           color="tab:gray")
    ax.bar(x + 0.2, a, width=0.4, label=f"After: iteration {after.iteration} ({after.experiment_id})",
           color="tab:blue")
    for i in range(len(METRICS)):
        delta = a[i] - b[i]
        ax.text(x[i], max(a[i], b[i]) + 0.03, f"{delta:+.3f}", ha="center", fontsize=11, fontweight="bold",
                color="tab:green" if delta >= 0 else "tab:red")
    ax.set_xticks(x, labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (best configuration)", fontsize=11)
    title = "Retrieval metrics before vs after" + (f": {change}" if change else "")
    ax.set_title("\n".join(textwrap.wrap(title, 70)), fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    return _save(fig, path)
