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

## Supported file types

| Type | Extensions | How it's processed |
|---|---|---|
| PDF | `.pdf` | Text per page, ruled tables as Markdown, embedded images (described if VLM is on) |
| Word | `.doc`, `.docx` | Converted to PDF with LibreOffice, then processed as PDF |
| PowerPoint | `.ppt`, `.pptx` | Converted to PDF; each slide is a page |
| Excel | `.xls`, `.xlsx`, `.xlsm` | Each sheet becomes a Markdown table; `.xls` is converted first |
| Image | `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp`, `.tif`, `.tiff` | Described by a vision model (`ENABLE_VLM=true` required) |

Office conversion needs LibreOffice installed (`soffice` on PATH, or set `LIBREOFFICE_BIN`).

## Modalities

Each chunk carries a `modality` (`text`, `table`, `image`), shown in citations so the model and the user know whether an answer came from prose, a table, or a figure. Tables are kept whole where they fit; oversized tables are split with the header repeated.

## Known limits

- **Table detection in PDFs** relies on ruling lines or clean alignment. Borderless tables are read as text.
- **Equations** are not extracted as LaTeX. The `equation` modality exists in the data model but has no producer yet. RAG-Anything's Mathematical Expression Parser would fill this gap.
- **Scanned pages** have no text layer and are skipped. Documents with no text at all are rejected. OCR is not included.
- **Images** are only searchable when a vision model is enabled.
- **Knowledge graph** indexing from RAG-Anything (LightRAG entities and relations) is not implemented. Retrieval is hybrid vector and keyword search over chunks.

## Knowledge graph (LightRAG-style)

At upload, each chunk goes through one LLM call that extracts entities and relations as JSON. The results form a graph. Each node and edge records the chunk ids it came from, so every graph fact can be traced to a passage.

**Query modes** (sent as `mode` on `POST /api/chat`):

| Mode | What it adds to chunk retrieval |
|---|---|
| `naive` | Nothing. Hybrid vector and keyword search with reranking. |
| `local` | Entities matching the question, their relations, and their source chunks. |
| `global` | Relations matching the question, and their source chunks. |
| `hybrid` | Both local and global. Default. |

Graph-derived chunks join the normal candidate pool and are reranked together with the search results, so the graph can add passages but can't bypass the reranker or the refusal threshold. Graph facts are shown to the model as non-citable background. Citations still point to chunks.

**Configuration:** `ENABLE_KG` (default `true`), `KG_PATH`, and `KG_WORKERS`. Each chunk is one extraction call, so a 100-chunk document costs about 100 LLM calls at upload. Set `ENABLE_KG=false` to skip graph building.

**Deletion** removes a document's chunks from every node and edge. Nodes and edges that no longer have sources are dropped.

**Testing:** `python -m eval.kg_smoke_test` covers parsing, provenance, the three retrieval modes, removal, and persistence. `python -m eval.api_smoke_test` runs the API end to end. Both use stub models, so they test logic, not extraction quality.

**Evaluating the graph:** `python -m eval.run_eval ... --kg --kg-chunk-size 800` builds a graph and reports retrieval recall for each graph mode next to the chunk-only baseline. Use `--answer --answer-mode hybrid` to score answers with the graph. The graph only earns its cost if it raises recall or answer accuracy on your question set.

**Limits:**
- Extraction quality depends on the model. A weaker model produces noisy entities, and duplicate names with different spellings can split a node. Normalization only ignores case and punctuation.
- Entity types are coarse, and the model picks them. Descriptions are concatenated, not summarized.
- Graph retrieval scores entities by embedding similarity of their name and description, not by exact match.
- The graph is stored as one JSON file and loaded into memory, which suits a single-user demo, not large corpora.
- Equations are still not extracted.

## Multimodal processing

**Typed content blocks.** Every parser returns blocks tagged text, table, image, or equation. The modality is stored with each chunk and shown in citations.

**Concurrent pipelines.** Ingestion runs two pipelines at once. The text pipeline chunks text and tables. The multimodal pipeline runs vision calls on figures and equations, then chunks their results. The chunks are merged in document order. Vision calls run with `VLM_WORKERS` concurrency and are retried on failure.

**Equation parser.** A text line is treated as an equation when it is mostly symbols and digits with few words (`E = mc^2`, `x_i = 5`). Equations are rendered as images. With `ENABLE_VLM=true`, the vision model transcribes each one to LaTeX, and the chunk holds both the LaTeX and the raw text. Without vision, the raw text is indexed. The heuristic is conservative, so it misses multi-line display formulas and formulas embedded as images in the PDF.

**VLM-enhanced queries.** Figures are stored as files (`IMAGE_DIR`). When vision is on and retrieved sources include figures, up to `VLM_QUERY_MAX_IMAGES` are attached to the prompt and the vision model answers from the actual image. The request's `vision` field overrides this per question: `true` or `false`, with `null` meaning use `ENABLE_VLM`. Text-only prompts still go to the chat model, so a Groq chat setup doesn't need a vision key unless figures are attached.

**Limits.**
- Figure and formula quality depends on the vision model. Nothing here has been checked against a real vision model.
- Figure bytes are kept on disk, so storage grows with the number of figures.
- Attached figures are sent to the vision provider on each question, which costs more per query than text.
- Equation detection is heuristic. Test it on your documents before relying on it.

## RAG modes

Choose one with `mode` on `POST /api/chat`:

| `mode` | What it does | Extra model calls per question |
|---|---|---|
| `traditional` | Retrieve, rerank, answer | 1 (the answer) |
| `crag` | Grades each passage as CORRECT, AMBIGUOUS or INCORRECT. If none are correct, rewrites the query once and retries. Refuses if still none. No web-search fallback. | 1 per passage, plus 1 rewrite and a second round if needed |
| `self_rag` | Filters passages for relevance, then checks the answer against its sources and regenerates once if unsupported. The retrieval-on-demand decision is not included. Not streamed. | 1 per passage, plus 1 check, plus 2 more if regenerated |
| `lightrag` | Graph retrieval over entities and relations. `lightrag_mode` is `local`, `global` or `hybrid`. | 0 for retrieval |
| `agentic` | One search with the question, then up to `AGENTIC_MAX_STEPS` rounds where the model decides whether to search again, then answers | 1 per round, plus 1 |
| `graphrag` | Answers from cluster summaries of the graph, plus their source chunks | 0 for retrieval; summaries are made at upload |

The old mode names `naive`, `local`, `global` and `hybrid` still work as aliases for `traditional` and `lightrag`.

Summaries for GraphRAG are made once per cluster at upload. Each cluster is cached by its node set, so a deletion only summarizes clusters that changed. Set `ENABLE_COMMUNITIES=false` to skip them.

## Retrieval techniques

Send any combination in `techniques`:

- `hybrid`: BM25 keyword search plus dense vector search, fused with reciprocal rank fusion. Without it, only dense search runs. The UI names it after the mode, for example `Hybrid Traditional RAG · BM25 keyword + dense vector`.
- `query_rewriting`: the model writes three alternative queries, and results for all of them are fused. One extra call per question.
- `contextual_compression`: keeps the two highest-scoring sentences of each passage, scored by the reranker.
- `parent_document`: searches small child chunks and returns their parent passage. Parents are groups of up to `PARENT_GROUP_SIZE` consecutive chunks from one page.
- `sentence_window`: returns the best sentence of each passage with two sentences of context on each side.
- `auto_merging`: where two or more children of one parent are retrieved, returns the parent instead.

`auto_merging` takes precedence over `parent_document`, and `contextual_compression` over `sentence_window`, when both are chosen.

## Rerankers

Send one or more in `rerankers`. Default: `cross_encoder`.

- `cross_encoder`: local, default. The only reranker whose score drives the refusal threshold.
- `bi_encoder`: cosine similarity of embeddings. Local, no threshold.
- `flashrank`: local. Install with `uv pip install flashrank`. Written against the package's documented API and tested with a stub. Not run against the real package in this build.
- `cohere`: hosted. Needs `COHERE_API_KEY` and `uv pip install cohere`. Sends passages to Cohere. Written against the documented API and tested with a stub, not against the live service.

When several rerankers are chosen, their rankings are fused with reciprocal rank fusion, and the refusal threshold uses the cross-encoder score if it is included.

`GET /api/options` lists every mode, technique and reranker with its availability, so the UI can offer only what is configured.

## Testing the pipeline

`python -m eval.pipeline_smoke_test` runs every mode, every technique and the reranker logic against stubs, including failure paths: CRAG refusal, Self-RAG regeneration, the agentic loop, missing Cohere key, and a missing graph.

## Limits of this version

- No web-search fallback for CRAG. The fallback is a query rewrite.
- Self-RAG always retrieves; it does not decide whether retrieval is needed.
- GraphRAG selects the few closest community summaries and does not run map-reduce over every community.
- LightRAG embeds the question directly rather than extracting high- and low-level keywords first.
- Contextual compression and sentence window score sentences with the reranker rather than an LLM.
- FlashRank and Cohere adapters are untested against the live libraries.
- Latency rises with the number of model calls per mode. Measure before enabling CRAG, Self-RAG or agentic mode on large documents.

## Machine learning evaluation set

The repository includes a corpus and 50 questions so the evaluation can run without writing your own data.

- **Corpus:** six short lecture-style documents (two pages each) in `eval/corpus_src/`, rendered to PDF by `python -m eval.build_corpus`. Topics: ML basics, regression, decision trees and ensembles, neural networks and backpropagation, CNNs, RNNs and transformers, and evaluation and training practice. Deep learning is covered in the neural network, CNN/RNN and practice documents.
- **Questions:** `eval/qa.jsonl`, written by `python -m eval.build_qa`. 50 questions: 36 single-passage (definition, fact, comparison, numeric), 6 multi-hop (each needs passages from two documents), and 8 unanswerable (the corpus doesn't cover them, so a good system refuses).
- **Labels:** each answerable question lists gold passages (`golds`: doc, page, snippet). `python -m eval.validate_qa` checks every snippet against its rendered page and every keyword against its gold passages, and fails on any mismatch.

**Schema change:** questions now carry `golds`, a list of passages, instead of one `gold_doc`/`gold_page`/`gold_snippet`. The old fields are still read. A multi-hop question counts as retrieved only when every passage is in the top k.

### Running it

```bash
./run_all.sh              # tests, build, validate, BM25 retrieval (no models, no keys)
./run_all.sh --models     # adds dense, hybrid and reranked retrieval (downloads models once)
./run_all.sh --answers    # adds answers and a comparison of modes (needs GROQ_API_KEY or MISTRAL_API_KEY)
```

Results go to `eval/results/`: `ablation.md` (retrieval, with recall by question type), `answer_eval.json` and `failures_<mode>.jsonl` (answers), and `compare.md` when several modes run.

### Reading the results

- **Label coverage** is the share of questions whose gold passages exist in some chunk. A coverage below 100% means chunking split an answer, so no retriever can find it at that chunk size. This is a limit of the chunker, not the retriever.
- **Recall by type** shows where a method fails. Multi-hop questions are the hardest, since they need two separate passages in the top k.
- **Accuracy** on answerable questions counts a question correct only if every answer keyword appears and the system did not refuse. Keyword matching is crude. Use a judge model or manual checks before quoting an accuracy figure.

### Known limits

- The corpus is small and written for this evaluation, so it tests whether a method works, not how it performs on real documents.
- Only one question is `numeric`. Most of the corpus is conceptual.
- Keyword answer matching can accept a wrong answer that happens to contain a keyword.
- Lexical-only runs measure BM25 retrieval. Answers need the embedding models and a chat model.
