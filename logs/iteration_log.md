## Iteration 1 → 2 (2026-10-22T22:31:25+00:00)

- **Change:** CHUNK_SIZE 512 → 256 tokens, OVERLAP 75 → 40 (all chunking methods)
- **Reason:** At 512 tokens 69–99% of chunks exceeded the embedding models' input limits (MiniLM 256, mpnet 384) and were silently truncated; dense retrieval was the weakest method (mean NDCG@5 0.36)
- **Best configuration:** recursive-minilm-hybrid → recursive-bm25

| Metric | Before | After | Delta |
|---|---|---|---|
| Recall@5 | 0.806 | 0.679 | -0.126 |
| Precision@5 | 0.250 | 0.200 | -0.050 |
| MRR | 0.794 | 0.660 | -0.135 |
| NDCG@5 | 0.731 | 0.588 | -0.142 |
| Hit@5 | 1.000 | 0.917 | -0.083 |

## Iteration 1 → 3 (2026-10-22T23:02:17+00:00)

- **Change:** Cross-encoder reranking (cross-encoder/ms-marco-MiniLM-L-6-v2): top 20 first-stage results reordered, cut to 5; 512-token chunks (iteration 2's 256-token chunks rejected)
- **Reason:** NDCG@5 (0.731) was the one retrieval target below its threshold (0.75); a cross-encoder scores query and chunk together, which should move relevant chunks up the top 5
- **Best configuration:** recursive-minilm-hybrid → recursive-bm25-rerank-cross-encoder

| Metric | Before | After | Delta |
|---|---|---|---|
| Recall@5 | 0.806 | 0.882 | +0.076 |
| Precision@5 | 0.250 | 0.283 | +0.033 |
| MRR | 0.794 | 0.861 | +0.067 |
| NDCG@5 | 0.731 | 0.831 | +0.100 |
| Hit@5 | 1.000 | 1.000 | +0.000 |

