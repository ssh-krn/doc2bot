// Support and evidence matrix for the About page.
//
// Two kinds of content, kept apart on purpose:
//  - Claims: what papers, benchmarks and vendors report. Taken from the sources listed in SOURCES. Not measured here.
//  - Model fit: what each model supports, costs, and how many calls each mode makes per question. Taken from
//    provider model cards and listings (noted per row), and from this codebase. Not accuracy measurements.
//
// No accuracy or latency numbers for these two models appear here, because none were measured in this build.

export type Strength = "High" | "Medium" | "Low" | "Contested" | "None found";

export interface MatrixRow {
  feature: string;
  kind: "Mode" | "Technique" | "Reranker" | "Baseline";
  claim: string;        // what the source reports
  check: string;        // independent or conflicting evidence, if any
  strength: Strength;
  groq: string;         // openai/gpt-oss-120b on Groq
  mistral: string;      // mistral-small-latest
  where: string;
}

export const MATRIX: MatrixRow[] = [
  {
    feature: "Traditional RAG", kind: "Baseline",
    claim: "No single claim. It is the reference the other rows are compared against.",
    check: "Every paper in this table compares against a version of it.",
    strength: "None found", groq: "1 call per question.", mistral: "1 call per question.",
    where: "pipeline/modes.py (traditional)",
  },
  {
    feature: "Hybrid search (BM25 + dense)", kind: "Technique",
    claim: "On BEIR, BM25 plus Contriever averages 43.8 nDCG, against 42.2 for BM25 alone and 34.9 for Contriever (ReDE-RF paper, arXiv 2410.21242). Elastic reports reciprocal rank fusion lifts NDCG@10 about 18% over BM25 on their setup.",
    check: "The gain depends on the dataset. Elastic's figure is from a vendor blog.",
    strength: "High", groq: "No model call. Retrieval runs on local embeddings, so the chat model has no effect.",
    mistral: "No model call. Same as Groq.", where: "retrieval.py, pipeline/candidates.py",
  },
  {
    feature: "Query rewriting (multi-query)", kind: "Technique",
    claim: "HyDE (Gao et al.) reports gains in zero-shot dense retrieval by searching with a generated hypothetical answer. Multi-query rewriting is a related but different method; the closest published evidence is HyDE.",
    check: "ReDE-RF (arXiv 2410.21242) finds real retrieved documents beat hypothetical ones, which is a point against generating answers to search with.",
    strength: "Low", groq: "+1 call per question, returns a JSON array. Reasoning tokens add latency.",
    mistral: "+1 call per question, returns a JSON array. JSON output is supported.",
    where: "pipeline/techniques.py (rewrite_queries)",
  },
  {
    feature: "Contextual compression", kind: "Technique",
    claim: "Vendor documentation (LangChain) describes it. No quantified evaluation located.",
    check: "No independent measurement found. This build scores sentences with the reranker, not an LLM, so the vendor description doesn't cover this version.",
    strength: "None found", groq: "No extra call. Reranker-scored.", mistral: "No extra call. Reranker-scored.",
    where: "pipeline/techniques.py (compress)",
  },
  {
    feature: "Parent document retriever", kind: "Technique",
    claim: "Vendor documentation (LangChain, LlamaIndex) describes small-child, large-parent retrieval. Quantified results not located.",
    check: "Indirect support: a study of chunk size (arXiv 2505.21700) finds chunk size is a major driver of retrieval quality, which is the trade-off this technique addresses.",
    strength: "Low", groq: "No extra call.", mistral: "No extra call.",
    where: "ingest.py (assign_parents), pipeline/techniques.py (parent_expand)",
  },
  {
    feature: "Sentence window retriever", kind: "Technique",
    claim: "LlamaIndex documents the approach as a way to add context around a matched sentence.",
    check: "Conflicting. The Chronicles of RAG study (arXiv 2401.07883, Table 9) reports sentence-window retrieval 23.8% worse on one configuration (gpt-3.5-turbo-1106) and gives hypotheses for why.",
    strength: "Contested", groq: "No extra call.", mistral: "No extra call.",
    where: "pipeline/techniques.py (sentence_window)",
  },
  {
    feature: "Auto-merging retriever", kind: "Technique",
    claim: "LlamaIndex documents the AutoMergingRetriever. No independent quantified evaluation located.",
    check: "None found.",
    strength: "None found", groq: "No extra call.", mistral: "No extra call.",
    where: "pipeline/techniques.py (auto_merge)",
  },
  {
    feature: "CRAG (Corrective RAG)", kind: "Mode",
    claim: "Original paper (arXiv 2401.15884) reports gains over standard RAG on a LLaMA2-7B-based generator: +19.0 points PopQA, +14.9 FactScore on Biography, +36.6 on PubHealth, +8.1 on ARC-Challenge. Those gains used a trained T5 evaluator and a web-search fallback.",
    check: "One independent reproduction (arXiv 2603.16169) with a different generator (Phi-3-mini) found 54.4 vs 51.4 on PopQA (+3.0) and 85.2 vs 84.8 on ARC (+0.4).",
    strength: "Medium", groq: "Calls per question: 1 grade per passage (about 5), then 1 answer. If nothing is correct, +1 rewrite, another round of grades, and an answer (up to about 13). Grading output is one word, but reasoning tokens still bill.",
    mistral: "Same call count. Grading is one word, which Mistral Small handles directly. Cheaper per call if the input price holds.",
    where: "pipeline/modes.py (crag)",
  },
  {
    feature: "Self-RAG", kind: "Mode",
    claim: "Trained 13B model (arXiv 2310.11511): PopQA 55.8, TriviaQA 69.3, PubHealth 74.5, ARC 73.1, Biography FactScore 80.2 (figures taken from a secondary summary; check the paper's tables). Human check on 50 samples found the reflection tokens mostly agree with annotators.",
    check: "The reflection tokens come from training, which this build doesn't do. The prompted version here uses the chat model as judge, so the paper's numbers don't apply directly.",
    strength: "Medium", groq: "Calls per question: about 5 relevance grades, 1 answer, 1 support check. If the check fails, +2. Roughly 7 to 9.",
    mistral: "Same call count. The support check needs the model to read sources and draft carefully, so answer quality here depends on how Mistral Small follows it. Unmeasured.",
    where: "pipeline/modes.py (self_rag_select), rag.py (_self_rag_answer)",
  },
  {
    feature: "LightRAG", kind: "Mode",
    claim: "The LightRAG paper (HKUDS) reports win rates of 60–85% over NaiveRAG across four datasets, judged by an LLM with GPT-4o-mini (range taken from a secondary summary; check the paper).",
    check: "An independent audit (arXiv 2506.06331) found NaiveRAG beat LightRAG in several settings (LightRAG win rate 39.06% on Agriculture). A study of Alzheimer's research (arXiv 2508.21238) found a plain LLM chatbot competitive, and LightRAG weaker on comprehensiveness and diversity in most cases.",
    strength: "Contested", groq: "0 retrieval calls. The answer is 1 call. Graph building at upload costs 1 call per chunk, which is the main cost.",
    mistral: "Same. Upload-time extraction on mistral-small-latest is 1 call per chunk.",
    where: "pipeline/modes.py (lightrag), kg/graph.py (context)",
  },
  {
    feature: "GraphRAG (Microsoft)", kind: "Mode",
    claim: "Vendor paper reports comprehensiveness win rates of 72–83% and diversity 62–82% against vector RAG on podcast and news corpora (figures from a secondary summary). Community summaries use up to 97% fewer tokens than the source text, per the same summaries.",
    check: "The audit (arXiv 2506.06331) confirms GraphRAG-style methods lead on comprehensiveness. It also finds LLM-judge win rates swing by more than 30 points with answer order, so headline rates overstate the gap.",
    strength: "Medium", groq: "Answer is 1 call. Summaries are 1 call per cluster, made at upload (up to 30).",
    mistral: "Same. Summaries are made with mistral-small-latest, so summary quality depends on that model.",
    where: "kg/graph.py (refresh_communities, community_context)",
  },
  {
    feature: "Agentic RAG", kind: "Mode",
    claim: "No single benchmark located. The category is described by vendors and surveys rather than one controlled evaluation.",
    check: "None found.", strength: "None found",
    groq: "2 to 4 calls per question (decisions plus answer). Each decision is a JSON reply.",
    mistral: "2 to 4 calls. Mistral Small's JSON output is supported, so the loop should parse. Unmeasured.",
    where: "pipeline/modes.py (agentic)",
  },
  {
    feature: "Cross-encoder reranker (default)", kind: "Reranker",
    claim: "BEIR (arXiv 2104.08663) found a 6-layer MiniLM cross-encoder performed best of 14 cross-attention rerankers tested on MS MARCO. NAIL (arXiv 2305.14499) notes cross-encoders are more accurate but more expensive than other rerankers.",
    check: "Strong in-domain. Out-of-domain results vary by dataset, which BEIR is designed to show.",
    strength: "High", groq: "No model call. Local. Adds latency per query, which is measured by the evaluation harness.",
    mistral: "No model call. Same as Groq.", where: "pipeline/rerank.py (score), store.py",
  },
  {
    feature: "Cohere Rerank 3.5", kind: "Reranker",
    claim: "Cohere reports improved relevance on multilingual and reasoning-style retrieval. A secondary write-up cites up to 25% better performance on hard retrieval tasks versus embedding search alone.",
    check: "A comparison by ZeroEntropy (a competitor) puts Cohere 3.5 at 0.709 nDCG@10 against 0.768 for their own model, and 172 ms against 150 ms on small payloads. Vendor-run; treat as one data point.",
    strength: "Low", groq: "Not a model call on Groq. It is one API call per question to Cohere, and passages leave the machine.",
    mistral: "Same as Groq: one Cohere API call per question, independent of the chat model.",
    where: "pipeline/rerank.py (_cohere_scores)",
  },
  {
    feature: "FlashRank", kind: "Reranker",
    claim: "Not evaluated in this research pass. No source located.",
    check: "The adapter is written against the documented API and tested with a stub. The real package hasn't been run.",
    strength: "None found", groq: "Local. No model call.", mistral: "Local. No model call.",
    where: "pipeline/rerank.py (_flashrank_scores)",
  },
  {
    feature: "Bi-encoder similarity", kind: "Reranker",
    claim: "Generally less accurate than cross-encoders, since question and passage are embedded separately. BEIR's comparison of models supports this ordering.",
    check: "Fast and cheap, which is the trade-off.",
    strength: "Medium", groq: "Local. No model call.", mistral: "Local. No model call.",
    where: "pipeline/rerank.py (score)",
  },
];

export interface ModelRow {
  capability: string;
  groq: string;
  mistral: string;
}

export const MODEL_FIT: ModelRow[] = [
  {
    capability: "Context window",
    groq: "131,072 tokens (Groq listings and third-party catalogs agree).",
    mistral: "128,000 tokens for Mistral Small 3.1 (model card). The API alias moves to newer versions, and one secondary source claims 256k; not confirmed in Mistral's docs.",
  },
  {
    capability: "Structured output (JSON)",
    groq: "Yes: JSON schema and tool calling, per several catalogs. Used by query rewriting, agentic decisions, and the grading prompts.",
    mistral: "Yes: JSON mode and JSON schema mode, per Mistral's API docs and the Small 3.1 card.",
  },
  {
    capability: "Reasoning",
    groq: "Reasoning model. Reasoning tokens are billed as output and add latency to every call, including one-word grades.",
    mistral: "Standard chat model in the Small 3.1 family. No reasoning step to pay for.",
  },
  {
    capability: "Image input",
    groq: "Text-only in most listings. One catalog lists image input. Treat as text-only. Figures are described and answered by the vision model setting, not this model.",
    mistral: "Yes for Mistral Small 3.1 (model card). Whether the API alias accepts images is unconfirmed; check before enabling figure features with this model.",
  },
  {
    capability: "Price per million tokens (input / output)",
    groq: "$0.15 / $0.60 in most catalogs. One lists $0.75 output. Check groq.com/pricing; this build doesn't fetch prices.",
    mistral: "A third-party page lists about $0.10 input. Not confirmed. Check mistral.ai/pricing.",
  },
];

export interface CallRow {
  mode: string;
  calls: string;
}

export const CALLS: CallRow[] = [
  { mode: "Traditional", calls: "1 (answer). +1 if query rewriting is on." },
  { mode: "CRAG", calls: "1 grade per passage (up to 5) + 1 answer = 6. If none correct: +1 rewrite, +5 grades, +1 answer = up to 13." },
  { mode: "Self-RAG", calls: "5 grades + 1 answer + 1 support check = 7. If unsupported: +1 answer +1 check = 9." },
  { mode: "LightRAG", calls: "1 (answer). +1 if query rewriting is on. Retrieval itself makes no call." },
  { mode: "Agentic", calls: "1 to 3 decisions + 1 answer = 2 to 4. Plus a rewrite per search if rewriting is on." },
  { mode: "GraphRAG", calls: "1 (answer). +1 if query rewriting is on. Summaries are made at upload, not per question." },
  { mode: "Upload (graph modes)", calls: "1 extraction per chunk, plus 1 summary per community (up to 30). Paid once per document." },
];

export const SOURCES: { label: string; url: string }[] = [
  { label: "CRAG, Yan et al. 2024 (arXiv 2401.15884)", url: "https://arxiv.org/abs/2401.15884" },
  { label: "Open-source CRAG reproduction (arXiv 2603.16169)", url: "https://arxiv.org/abs/2603.16169" },
  { label: "Self-RAG, Asai et al. 2023 (arXiv 2310.11511)", url: "https://arxiv.org/abs/2310.11511" },
  { label: "Independent audit of GraphRAG and LightRAG evaluation (arXiv 2506.06331)", url: "https://arxiv.org/abs/2506.06331" },
  { label: "LLM-based GraphRAG study on Alzheimer's research (arXiv 2508.21238)", url: "https://arxiv.org/abs/2508.21238" },
  { label: "BEIR benchmark (arXiv 2104.08663)", url: "https://arxiv.org/abs/2104.08663" },
  { label: "NAIL, lexical indices and cross-encoders (arXiv 2305.14499)", url: "https://arxiv.org/abs/2305.14499" },
  { label: "ReDE-RF, zero-shot retrieval with relevance feedback (arXiv 2410.21242)", url: "https://arxiv.org/abs/2410.21242" },
  { label: "The Chronicles of RAG (arXiv 2401.07883)", url: "https://arxiv.org/abs/2401.07883" },
  { label: "Chunk size for long-document retrieval (arXiv 2505.21700)", url: "https://arxiv.org/abs/2505.21700" },
  { label: "Mistral Small 3.1 model card", url: "https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503" },
  { label: "Elastic, hybrid retrieval with reciprocal rank fusion", url: "https://www.elastic.co/search-labs/blog/improving-information-retrieval-elastic-stack-hybrid" },
];
