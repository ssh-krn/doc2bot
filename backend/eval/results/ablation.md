| Chunk size | Method | Rerank | Recall@1 | Recall@3 | Recall@5 | MRR@5 | p50 ms | p95 ms | Label coverage |
|---|---|---|---|---|---|---|---|---|---|
| 300 | bm25 | no | 50% | 69% | 71% | 0.69 | 0.1 | 0.2 | 76% |
| 600 | bm25 | no | 67% | 83% | 83% | 0.81 | 0.2 | 0.3 | 88% |
| 900 | bm25 | no | 79% | 93% | 95% | 0.94 | 0.1 | 0.2 | 100% |


Recall@5 by question type:

| Chunk size | Method | fact | definition | comparison | numeric | multi_hop |
|---|---|---|---|---|---|---|
| 300 | bm25 | 86% | 54% | 100% | 100% | 33% |
| 600 | bm25 | 86% | 92% | 75% | 100% | 67% |
| 900 | bm25 | 100% | 92% | 100% | 100% | 83% |
