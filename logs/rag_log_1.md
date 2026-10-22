# Evaluation run 2026-10-22T22:18:04+00:00 (iteration 1, 15 configurations, generator openai/gpt-6-luna)

- **Ground truth:** chunks overlapping the qrels section (word 3-grams, either direction, ≥ 50%)
- **Best configuration** (NDCG@5, then MRR): **recursive-minilm-hybrid**
- **Judge:** deepseek/deepseek-v4-flash

| Configuration | Recall@5 | Precision@5 (max) | MRR | NDCG@5 | Hit@5 | Paper P@5 | Judge overall | Retrieval ms | End-to-end s |
|---|---|---|---|---|---|---|---|---|---|
| recursive-minilm-hybrid | 0.806 | 0.250 (0.35) | 0.794 | 0.731 | 1.000 | 0.967 | 4.583 | 5.427 | 2.898 |
| recursive-bm25 | 0.771 | 0.217 (0.35) | 0.736 | 0.681 | 1.000 | 0.933 | 4.458 | 6.378 | 2.588 |
| recursive-mpnet-hybrid | 0.833 | 0.267 (0.35) | 0.674 | 0.672 | 1.000 | 0.967 | 4.812 | 5.597 | 2.984 |
| recursive-minilm-dense | 0.729 | 0.233 (0.35) | 0.593 | 0.576 | 0.917 | 0.917 | 4.375 | 0.108 | 2.292 |
| semantic-bm25 | 0.720 | 0.200 (0.37) | 0.531 | 0.549 | 0.833 | 0.933 | 4.500 | 5.638 | 2.698 |
| fixed-bm25 | 0.722 | 0.200 (0.33) | 0.517 | 0.538 | 0.833 | 0.967 | 4.438 | 5.306 | 2.982 |
| semantic-mpnet-hybrid | 0.732 | 0.233 (0.37) | 0.451 | 0.505 | 0.833 | 0.917 | 4.500 | 7.197 | 3.246 |
| recursive-mpnet-dense | 0.556 | 0.200 (0.35) | 0.521 | 0.485 | 0.750 | 0.917 | 4.375 | 0.210 | 3.803 |
| semantic-minilm-hybrid | 0.637 | 0.200 (0.37) | 0.444 | 0.465 | 0.750 | 0.917 | 4.479 | 6.764 | 3.080 |
| fixed-mpnet-hybrid | 0.576 | 0.183 (0.33) | 0.451 | 0.436 | 0.667 | 0.933 | 4.479 | 5.436 | 3.188 |
| fixed-minilm-hybrid | 0.576 | 0.183 (0.33) | 0.403 | 0.427 | 0.667 | 0.950 | 4.417 | 5.677 | 3.007 |
| semantic-minilm-dense | 0.406 | 0.167 (0.37) | 0.342 | 0.324 | 0.583 | 0.883 | 4.333 | 0.137 | 3.379 |
| semantic-mpnet-dense | 0.315 | 0.150 (0.37) | 0.312 | 0.275 | 0.417 | 0.917 | 4.042 | 0.279 | 3.283 |
| fixed-mpnet-dense | 0.299 | 0.117 (0.33) | 0.333 | 0.274 | 0.417 | 0.933 | 4.542 | 20.947 | 2.686 |
| fixed-minilm-dense | 0.368 | 0.133 (0.33) | 0.196 | 0.229 | 0.500 | 0.917 | 4.229 | 8.627 | 2.508 |

**Best configuration judge scores:** relevance 4.83, accuracy 4.50, completeness 4.42, citation_quality 4.58, overall 4.58

# Evaluation run 2026-10-22T22:18:33+00:00 (iteration 1, 15 configurations, generator openai/gpt-6-luna)

- **Ground truth:** chunks overlapping the qrels section (word 3-grams, either direction, ≥ 50%)
- **Best configuration** (NDCG@5, then MRR): **recursive-minilm-hybrid**
- **Judge:** deepseek/deepseek-v4-flash

| Configuration | Recall@5 | Precision@5 (max) | MRR | NDCG@5 | Hit@5 | Paper P@5 | Judge overall | Retrieval ms | End-to-end s |
|---|---|---|---|---|---|---|---|---|---|
| recursive-minilm-hybrid | 0.806 | 0.250 (0.35) | 0.794 | 0.731 | 1.000 | 0.967 | 4.583 | 15.006 | 2.908 |
| recursive-bm25 | 0.771 | 0.217 (0.35) | 0.736 | 0.681 | 1.000 | 0.933 | 4.458 | 4.443 | 2.586 |
| recursive-mpnet-hybrid | 0.833 | 0.267 (0.35) | 0.674 | 0.672 | 1.000 | 0.967 | 4.812 | 28.078 | 3.007 |
| recursive-minilm-dense | 0.729 | 0.233 (0.35) | 0.593 | 0.576 | 0.917 | 0.917 | 4.375 | 7.000 | 2.299 |
| semantic-bm25 | 0.720 | 0.200 (0.37) | 0.531 | 0.549 | 0.833 | 0.933 | 4.500 | 5.999 | 2.699 |
| fixed-bm25 | 0.722 | 0.200 (0.33) | 0.517 | 0.538 | 0.833 | 0.967 | 4.438 | 4.374 | 2.982 |
| semantic-mpnet-hybrid | 0.732 | 0.233 (0.37) | 0.451 | 0.505 | 0.833 | 0.917 | 4.500 | 27.333 | 3.268 |
| recursive-mpnet-dense | 0.556 | 0.200 (0.35) | 0.521 | 0.485 | 0.750 | 0.917 | 4.375 | 17.869 | 3.821 |
| semantic-minilm-hybrid | 0.637 | 0.200 (0.37) | 0.444 | 0.465 | 0.750 | 0.917 | 4.479 | 17.784 | 3.091 |
| fixed-mpnet-hybrid | 0.576 | 0.183 (0.33) | 0.451 | 0.436 | 0.667 | 0.933 | 4.479 | 26.966 | 3.216 |
| fixed-minilm-hybrid | 0.576 | 0.183 (0.33) | 0.403 | 0.427 | 0.667 | 0.950 | 4.417 | 15.159 | 3.016 |
| semantic-minilm-dense | 0.406 | 0.167 (0.37) | 0.342 | 0.324 | 0.583 | 0.883 | 4.333 | 7.105 | 3.386 |
| semantic-mpnet-dense | 0.315 | 0.150 (0.37) | 0.312 | 0.275 | 0.417 | 0.917 | 4.042 | 18.873 | 3.305 |
| fixed-mpnet-dense | 0.299 | 0.117 (0.33) | 0.333 | 0.274 | 0.417 | 0.933 | 4.542 | 20.135 | 2.693 |
| fixed-minilm-dense | 0.368 | 0.133 (0.33) | 0.196 | 0.229 | 0.500 | 0.917 | 4.229 | 8.419 | 2.513 |

**Best configuration judge scores:** relevance 4.83, accuracy 4.50, completeness 4.42, citation_quality 4.58, overall 4.58

