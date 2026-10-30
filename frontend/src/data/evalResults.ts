// Evaluation results written by the backend harness (backend/eval/results/).
// Copy these files into this folder to show them on the About page:
//   ablation.json     <- backend/eval/results/ablation.json
//   answer_eval.json  <- backend/eval/results/answer_eval.json
// Vite loads any JSON placed here. If a file is missing, the About page says the results aren't measured yet.

export interface AblationRow {
  chunk_size: number;
  mode: string;           // e.g. dense | bm25 | hybrid | kg-local | kg-global | kg-hybrid
  rerank: boolean;
  label_coverage: number; // share of questions whose gold passage exists in at least one chunk
  "recall@1": number;
  "recall@3": number;
  "recall@5": number;
  "mrr@5": number;
  p50_ms: number;
  p95_ms: number;
}

export interface AnswerSummary {
  chunk_size: number;
  questions: number;
  answer_accuracy_answerable: number;
  refusal_accuracy_unanswerable: number;
  citation_error_rate: number;
  retrieval_p50_ms: number;
  llm_p50_ms: number;
  llm_p95_ms: number;
  mean_input_tokens: number;
  mean_output_tokens: number;
  cost_per_query_usd: number;
  cost_per_1000_queries_usd: number;
}

const files = import.meta.glob("./*.json", { eager: true, import: "default" }) as Record<string, unknown>;

export const ablationRows = (files["./ablation.json"] as AblationRow[] | undefined) ?? null;

export const answerSummary =
  ((files["./answer_eval.json"] as { summary: AnswerSummary } | undefined) ?? null)?.summary ?? null;
