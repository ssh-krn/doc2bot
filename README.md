# PDF RAG with hybrid retrieval, reranking, and an evaluation harness

Ask questions over your PDFs and get cited answers. Refuses when the documents don't support an answer.

## Design

- **Ingestion.** PyMuPDF extracts text per page, then a recursive splitter chunks it. Each chunk keeps its filename and page, so every citation points to a real page.
- **Hybrid retrieval.** Dense vectors (BGE-small, local) and BM25 keyword search each return candidates. Reciprocal rank fusion merges them. Dense search handles paraphrase; BM25 handles exact terms like clause numbers and product codes.
- **Reranking.** A local cross-encoder rescores the fused candidates. It costs latency, and the evaluation measures whether it's worth it.
- **Refusal.** If the best reranked passage scores below `RERANK_MIN_SCORE`, the system refuses rather than letting the LLM guess from weak evidence.
- **LLM.** Groq or Mistral through LangChain, at temperature 0, with numbered sources the model must cite.

## Setup

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.lock.txt
cp .env.example .env          # add GROQ_API_KEY or MISTRAL_API_KEY
uvicorn app.main:app --port 8000
```

Offline check, no models or API keys needed: `python -m eval.smoke_test`

## Evaluation

### 1. Build the question set

Put 5–15 PDFs in `eval/corpus/`. Write 50–100 questions in `eval/qa.jsonl`, one JSON object per line:

```json
{"id": "q001", "question": "How many days is the refund window?", "gold_doc": "policy.pdf", "gold_page": 4, "gold_snippet": "within 30 days of purchase", "gold_answer_keywords": ["30 days"], "answerable": true}
{"id": "q002", "question": "What is the CEO's salary?", "answerable": false}
```

- `gold_snippet`: copy 8–25 words exactly from the page. Matching is case- and whitespace-insensitive.
- `gold_answer_keywords`: facts a correct answer must contain.
- `answerable: false`: questions the documents can't answer. These measure refusal.
- Include hard cases: tables, multi-page answers, paraphrased questions, and questions that share keywords with a different section.

### 2. Run the ablations (no LLM calls)

```bash
python -m eval.run_eval --corpus eval/corpus --qa eval/qa.jsonl --chunk-sizes 400 800 1200
```

This compares dense, BM25, and hybrid retrieval, each with and without reranking, across chunk sizes. Results go to `eval/results/ablation.md`.

**Label coverage** is the share of questions whose gold snippet exists in at least one chunk. If it's below 100% at a given chunk size, the chunker split an answer across boundaries, and no retriever can find it at that size.

### 3. Run end-to-end answer evaluation (calls the LLM)

```bash
export PRICE_IN_PER_M=...     # set from your provider's current pricing page
export PRICE_OUT_PER_M=...
python -m eval.run_eval --corpus eval/corpus --qa eval/qa.jsonl --answer --answer-chunk-size 800
```

This writes `eval/results/answer_eval.json` (summary and per-question results) and `eval/results/failures.jsonl`, which classifies each failure:

| Category | Meaning |
|---|---|
| `retrieval_miss` | Gold passage exists but wasn't retrieved |
| `snippet_split_by_chunking` | Gold answer was split across chunks, so it can't be retrieved |
| `wrong_answer` | Right passage retrieved, answer missing required facts |
| `false_refusal` | Answerable question refused |
| `bad_citation` | Answer cites a source that doesn't support the claim |
| `hallucinated_answer` | Unanswerable question answered instead of refused |

## Results

*Not yet measured. Fill these in from `eval/results/` after running on your corpus. Don't publish numbers you didn't measure.*

### Retrieval ablation

[paste eval/results/ablation.md here]

### End-to-end

| Metric | Value |
|---|---|
| Questions (answerable / unanswerable) | |
| Answer accuracy (answerable) | |
| Refusal accuracy (unanswerable) | |
| Citation error rate | |
| Retrieval p50 / LLM p50 / LLM p95 | |
| Mean input / output tokens per query | |
| Cost per query (hosted API, at stated prices) | |

### Failure analysis

[For each category in failures.jsonl, write 1–2 example questions, the root cause, and what you changed or would change.]

## Limitations

- Scanned PDFs are rejected because there's no OCR. Pages without a text layer are skipped.
- BM25 has no stemming, so "require" doesn't match "requires". The smoke test shows this.
- Answer correctness uses keyword matching, which misses correct paraphrases and can accept answers that include a keyword incorrectly. A stronger grader would use a judge model and a human spot check.
- The evaluation set is small and written by one person, so results have wide confidence intervals. Report the question count next to every percentage.
- Latency numbers depend on hardware. Record the machine (CPU/GPU, RAM) with any result.
- Document state is in memory plus local Qdrant. It's fine for one process, not for multiple workers.
