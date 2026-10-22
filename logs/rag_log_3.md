# Evaluation run 2026-10-22T23:02:16+00:00 (iteration 3, 15 configurations, generator openai/gpt-6-luna)

- **Ground truth:** chunks overlapping the qrels section (word 3-grams, either direction, ≥ 50%)
- **Best configuration** (NDCG@5, then MRR): **recursive-bm25-rerank-cross-encoder**
- **Judge:** deepseek/deepseek-v4-flash

| Configuration | Recall@5 | Precision@5 (max) | MRR | NDCG@5 | Hit@5 | Paper P@5 | Judge overall | Retrieval ms | End-to-end s |
|---|---|---|---|---|---|---|---|---|---|
| recursive-bm25-rerank-cross-encoder | 0.882 | 0.283 (0.35) | 0.861 | 0.831 | 1.000 | 1.000 | 4.729 | 112.556 | 3.023 |
| recursive-mpnet-dense-rerank-cross-encoder | 0.882 | 0.283 (0.35) | 0.861 | 0.831 | 1.000 | 0.967 | 4.708 | 122.237 | 3.256 |
| recursive-mpnet-hybrid-rerank-cross-encoder | 0.882 | 0.283 (0.35) | 0.861 | 0.831 | 1.000 | 1.000 | 4.688 | 129.033 | 2.966 |
| recursive-minilm-dense-rerank-cross-encoder | 0.840 | 0.267 (0.35) | 0.861 | 0.799 | 1.000 | 0.950 | 4.771 | 113.312 | 2.652 |
| recursive-minilm-hybrid-rerank-cross-encoder | 0.840 | 0.267 (0.35) | 0.861 | 0.799 | 1.000 | 0.983 | 4.604 | 120.016 | 2.602 |
| fixed-mpnet-dense-rerank-cross-encoder | 0.875 | 0.267 (0.33) | 0.718 | 0.698 | 1.000 | 0.917 | 4.750 | 125.877 | 2.879 |
| fixed-minilm-hybrid-rerank-cross-encoder | 0.875 | 0.267 (0.33) | 0.711 | 0.692 | 1.000 | 1.000 | 4.542 | 120.312 | 3.329 |
| fixed-mpnet-hybrid-rerank-cross-encoder | 0.875 | 0.267 (0.33) | 0.711 | 0.692 | 1.000 | 0.983 | 4.688 | 128.746 | 2.311 |
| fixed-bm25-rerank-cross-encoder | 0.854 | 0.250 (0.33) | 0.711 | 0.679 | 1.000 | 1.000 | 4.792 | 113.606 | 2.621 |
| semantic-mpnet-hybrid-rerank-cross-encoder | 0.734 | 0.250 (0.37) | 0.554 | 0.575 | 0.833 | 0.950 | 4.354 | 130.847 | 2.742 |
| semantic-bm25-rerank-cross-encoder | 0.706 | 0.233 (0.37) | 0.554 | 0.558 | 0.833 | 0.967 | 4.500 | 113.850 | 3.048 |
| semantic-minilm-hybrid-rerank-cross-encoder | 0.706 | 0.233 (0.37) | 0.554 | 0.558 | 0.833 | 0.983 | 4.667 | 121.403 | 3.471 |
| semantic-mpnet-dense-rerank-cross-encoder | 0.692 | 0.233 (0.37) | 0.554 | 0.550 | 0.833 | 0.900 | 4.229 | 122.017 | 3.219 |
| fixed-minilm-dense-rerank-cross-encoder | 0.625 | 0.217 (0.33) | 0.586 | 0.537 | 0.750 | 0.950 | 4.542 | 114.399 | 2.938 |
| semantic-minilm-dense-rerank-cross-encoder | 0.526 | 0.200 (0.37) | 0.458 | 0.438 | 0.667 | 0.933 | 4.271 | 112.888 | 2.894 |

**Best configuration judge scores:** relevance 5.00, accuracy 4.83, completeness 4.58, citation_quality 4.50, overall 4.73

