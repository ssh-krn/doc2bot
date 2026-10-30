// Descriptions of the system. Every claim here matches what the backend implements. Where a feature is a
// simplified version of the published method, the entry says so.

export interface Item {
  name: string;
  what: string;
  why: string;
  where: string;
  limits?: string;
}

export const WORKFLOW: { title: string; body: string }[] = [
  {
    title: "Upload",
    body: "You upload a PDF, Word, PowerPoint, Excel or image file. The server checks the type and size before any processing.",
  },
  {
    title: "Parse into typed blocks",
    body: "A format-specific parser turns the file into typed blocks: text, tables, figures and equations. Office files are converted with LibreOffice first. Each spreadsheet sheet becomes a table.",
  },
  {
    title: "Process text and figures in parallel",
    body: "Two pipelines run at once. The text pipeline chunks text by page with overlap and keeps tables whole. The multimodal pipeline, when vision is enabled, describes figures and transcribes equations to LaTeX. Chunks are grouped into parents for the parent-document and auto-merging retrievers.",
  },
  {
    title: "Index for search",
    body: "Each chunk is embedded for vector search and added to a keyword index. Both are stored locally. Only the question and the passages sent to the chat model leave the machine, and only if you use a hosted model.",
  },
  {
    title: "Build the knowledge graph and communities",
    body: "The chat model extracts entities and relations from each chunk. Code merges them into a graph that records which chunks each fact came from. The graph is then clustered into communities, and each community gets a summary.",
  },
  {
    title: "Choose a pipeline",
    body: "The RAG mode and the retrieval techniques you pick decide how candidates are found. Some modes make extra model calls to grade, rewrite or check, and the pipeline notes show each step.",
  },
  {
    title: "Find and rank candidates",
    body: "Candidates come from chunk search (keyword plus vector, or vector only), and in graph modes also from graph facts or community summaries. One or more rerankers score them against the question.",
  },
  {
    title: "Expand, compress and gate",
    body: "Parent, merge, compression or window techniques reshape the passages. Passages whose cross-encoder score is below the threshold are dropped, and if nothing is left the system refuses rather than guessing.",
  },
  {
    title: "Answer with citations",
    body: "The chat model answers from numbered sources with inline citations. Retrieved figures are attached when vision is on. The answer streams in, with sources, notes and graph facts arriving first.",
  },
];

export const MODES: Item[] = [
  {
    name: "Traditional RAG",
    what: "Retrieve passages once, rerank them, and answer from them in one model call.",
    why: "The baseline. Fast and predictable, and it is the reference the other modes are measured against.",
    where: "backend/app/pipeline/modes.py (traditional)",
  },
  {
    name: "CRAG (Corrective RAG)",
    what: "Grades each retrieved passage as CORRECT, AMBIGUOUS or INCORRECT with the chat model. Keeps the correct ones. If none are correct, rewrites the query once and retries. If still none, refuses.",
    why: "Stops the model from answering from passages that don't help, and recovers when the first query was badly phrased.",
    where: "backend/app/pipeline/modes.py (crag)",
    limits: "The published method falls back to web search. This app has no web search, so the fallback is a query rewrite.",
  },
  {
    name: "Self-RAG",
    what: "Filters passages for relevance, then checks the drafted answer against its sources. If the answer is not fully supported, it is regenerated once with instructions to remove unsupported claims.",
    why: "Catches answers that go beyond their sources, which retrieval alone can't prevent.",
    where: "backend/app/pipeline/modes.py (self_rag_select), backend/app/rag.py (_self_rag_answer)",
    limits: "The published method also decides whether to retrieve at all. This app always retrieves, since it only answers from its documents. Answers are not streamed, because the check needs the full draft.",
  },
  {
    name: "LightRAG",
    what: "Searches the knowledge graph for entities (local), relations (global), or both (hybrid), adds the graph facts to the prompt, and includes the source chunks of the matching facts in the candidate pool.",
    why: "Connects facts that sit in different passages, which chunk similarity alone can miss.",
    where: "backend/app/pipeline/modes.py (lightrag), backend/app/kg/graph.py (context)",
    limits: "The published method first asks the model to turn the question into high- and low-level keywords. This app embeds the question directly, which costs one model call less but may miss on vague questions.",
  },
  {
    name: "Agentic RAG",
    what: "Runs one search with the question, then lets the chat model decide, for up to three rounds, whether to search again with a new query. Then answers from everything gathered.",
    why: "Multi-part questions often need a second search that the first query didn't suggest.",
    where: "backend/app/pipeline/modes.py (agentic)",
    limits: "Each round is one more model call. The loop stops early if the model asks for a query it already used.",
  },
  {
    name: "GraphRAG",
    what: "Clusters the knowledge graph with the Louvain method, summarizes each cluster with the chat model, and answers from the summaries closest to the question, plus the chunks behind them.",
    why: "Broad questions ('what are the main themes') are answered better from cluster summaries than from individual passages.",
    where: "backend/app/kg/graph.py (refresh_communities, community_context), backend/app/kg/build.py (llm_summarize_factory)",
    limits: "The published method also runs a map-reduce step over every community. This app selects the closest few summaries instead. Summaries are cached by cluster, so a deletion only re-summarizes clusters that changed.",
  },
];

export const TECHNIQUES: Item[] = [
  {
    name: "Hybrid search",
    what: "BM25 keyword search and dense vector search each rank the chunks. Their rankings are fused with reciprocal rank fusion.",
    why: "Vectors match paraphrases and keywords match exact terms such as clause numbers and names. Each one misses what the other finds.",
    where: "backend/app/retrieval.py, backend/app/pipeline/candidates.py",
    limits: "Named by the RAG type it runs under. For example, Hybrid Traditional RAG uses BM25 keyword and dense vector search, while Hybrid LightRAG also searches entity and relation vectors.",
  },
  {
    name: "Query rewriting",
    what: "The chat model writes three alternative phrasings of the question. Each is searched, and the results are fused.",
    why: "A question phrased differently from the document often misses the passage that answers it.",
    where: "backend/app/pipeline/techniques.py (rewrite_queries)",
    limits: "One extra model call per question.",
  },
  {
    name: "Contextual compression",
    what: "For each passage, keeps the two sentences the reranker scores highest, in their original order.",
    why: "A long passage can contain one useful sentence among many irrelevant ones. Keeping only the useful sentences shortens the prompt.",
    where: "backend/app/pipeline/techniques.py (compress)",
    limits: "This version scores sentences with the reranker. The published method usually uses an LLM to extract them, which can follow context better but costs more.",
  },
  {
    name: "Parent document retriever",
    what: "Chunks are indexed in small children. When a child matches, the answer uses its larger parent passage (up to three children from one page).",
    why: "Small chunks match questions precisely, and a larger passage gives the model enough surrounding context.",
    where: "backend/app/ingest.py (assign_parents), backend/app/pipeline/techniques.py (parent_expand)",
  },
  {
    name: "Sentence window retriever",
    what: "Finds the best-scoring sentence in each passage and returns it with two sentences of context on each side.",
    why: "A single sentence is precise, and the window keeps enough context for it to make sense.",
    where: "backend/app/pipeline/techniques.py (sentence_window)",
    limits: "Sentences are scored by the reranker, not embedded separately, so the window is chosen from the passage already retrieved.",
  },
  {
    name: "Auto-merging retriever",
    what: "If two or more children of the same parent are retrieved, the parent passage replaces them. A lone child stays as it is.",
    why: "Several hits in one section suggest the whole section is relevant, which the parent shows better than the scattered children.",
    where: "backend/app/pipeline/techniques.py (auto_merge)",
  },
];

export const RERANKERS: Item[] = [
  {
    name: "Cross-encoder (default)",
    what: "A local model that reads each question and passage together and scores their relevance.",
    why: "Much more accurate than comparing separate embeddings. It is the only reranker whose score drives the refusal threshold.",
    where: "backend/app/store.py, backend/app/pipeline/rerank.py",
  },
  {
    name: "FlashRank",
    what: "A lightweight local reranker for CPU use.",
    why: "A faster local option when the cross-encoder is too slow.",
    where: "backend/app/pipeline/rerank.py (_flashrank_scores)",
    limits: "Written against the package's documented interface and tested with a stub. Not run against the real package in this build.",
  },
  {
    name: "Cohere Rerank",
    what: "A hosted reranking API.",
    why: "Strong ranking quality with no local model to run.",
    where: "backend/app/pipeline/rerank.py (_cohere_scores)",
    limits: "Sends the question and passages to Cohere. Needs COHERE_API_KEY. Written against the documented API and tested with a stub, not run against the live service.",
  },
  {
    name: "Bi-encoder similarity",
    what: "Cosine similarity between the question's embedding and each passage's embedding.",
    why: "Fast and needs no extra model. Useful for comparison against the cross-encoder.",
    where: "backend/app/pipeline/rerank.py (score)",
    limits: "Coarser than a cross-encoder. Its scores don't drive the refusal threshold.",
  },
];

export const METHODS_NOTE =
  "Several reranker choices can be combined. Their rankings are fused, and the refusal threshold still uses only the cross-encoder score.";
