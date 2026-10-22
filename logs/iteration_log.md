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

