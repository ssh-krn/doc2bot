// Evaluation results for the About page. Source: backend/eval/results/ (run_eval).
// ablation.json: retrieval rows. Fields that were not measured are null and show as "–".
// answer_eval.json: end-to-end answers. Only Traditional RAG has been run so far.
// To update, copy the new files from backend/eval/results/ into this folder and rebuild.

export interface AblationRow {
  chunk_size: number;
  mode: string;            // dense | bm25 | hybrid
  rerank: boolean;
  label_coverage: number;  // share of questions whose gold passages exist in some chunk
  "recall@1": number | null;
  "recall@3": number | null;
  "recall@5": number | null;
  "mrr@5": number | null;
  p50_ms: number | null;
  p95_ms: number | null;
}

export interface AnswerSummary {
  mode: string;
  chunk_size: number;
  questions: number;
  answer_accuracy_answerable: number;
  refusal_accuracy_unanswerable: number;
  answer_accuracy_by_type?: Record<string, number>;
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
