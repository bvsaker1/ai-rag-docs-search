# Evaluation run 2026-10-22T22:31:24+00:00 (iteration 2, 15 configurations, generator openai/gpt-6-luna)

- **Ground truth:** chunks overlapping the qrels section (word 3-grams, either direction, ≥ 50%)
- **Best configuration** (NDCG@5, then MRR): **recursive-bm25**
- **Judge:** deepseek/deepseek-v4-flash

| Configuration | Recall@5 | Precision@5 (max) | MRR | NDCG@5 | Hit@5 | Paper P@5 | Judge overall | Retrieval ms | End-to-end s |
|---|---|---|---|---|---|---|---|---|---|
| recursive-bm25 | 0.679 | 0.200 (0.47) | 0.660 | 0.588 | 0.917 | 0.933 | 4.729 | 8.595 | 2.298 |
| recursive-minilm-hybrid | 0.703 | 0.233 (0.47) | 0.667 | 0.577 | 1.000 | 0.967 | 4.438 | 20.533 | 2.477 |
| recursive-mpnet-hybrid | 0.703 | 0.233 (0.47) | 0.637 | 0.565 | 1.000 | 0.967 | 4.729 | 30.183 | 2.527 |
| semantic-mpnet-hybrid | 0.617 | 0.250 (0.50) | 0.660 | 0.543 | 0.917 | 0.950 | 4.583 | 29.403 | 3.386 |
| semantic-bm25 | 0.592 | 0.200 (0.50) | 0.646 | 0.533 | 0.833 | 0.917 | 4.396 | 8.742 | 3.039 |
| semantic-minilm-hybrid | 0.605 | 0.233 (0.50) | 0.625 | 0.524 | 0.917 | 0.950 | 4.812 | 20.290 | 3.153 |
| semantic-minilm-dense | 0.511 | 0.200 (0.50) | 0.662 | 0.481 | 0.833 | 0.917 | 4.208 | 7.133 | 4.019 |
| recursive-minilm-dense | 0.558 | 0.217 (0.47) | 0.551 | 0.476 | 0.833 | 0.933 | 4.833 | 6.806 | 2.328 |
| fixed-minilm-hybrid | 0.542 | 0.217 (0.52) | 0.576 | 0.476 | 0.750 | 0.983 | 4.312 | 19.245 | 2.204 |
| fixed-minilm-dense | 0.597 | 0.250 (0.52) | 0.482 | 0.454 | 0.833 | 0.967 | 4.042 | 9.542 | 2.760 |
| recursive-mpnet-dense | 0.482 | 0.217 (0.47) | 0.590 | 0.454 | 0.750 | 0.933 | 4.646 | 17.737 | 2.715 |
| fixed-mpnet-hybrid | 0.542 | 0.217 (0.52) | 0.500 | 0.445 | 0.750 | 0.967 | 4.604 | 29.476 | 2.500 |
| semantic-mpnet-dense | 0.450 | 0.217 (0.50) | 0.549 | 0.439 | 0.750 | 0.917 | 3.875 | 19.140 | 2.493 |
| fixed-bm25 | 0.493 | 0.183 (0.52) | 0.444 | 0.391 | 0.750 | 0.933 | 4.229 | 8.671 | 3.118 |
| fixed-mpnet-dense | 0.458 | 0.200 (0.52) | 0.438 | 0.358 | 0.750 | 0.933 | 4.104 | 20.134 | 3.206 |

**Best configuration judge scores:** relevance 5.00, accuracy 4.83, completeness 4.33, citation_quality 4.75, overall 4.73

