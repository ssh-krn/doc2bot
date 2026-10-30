import { ablationRows, answerSummary, type AblationRow, type AnswerSummary } from "../data/evalResults";

const pct = (v: number) => `${(v * 100).toFixed(0)}%`;
const ms = (v: number) => `${Math.round(v)} ms`;

const MODE_LABELS: Record<string, string> = {
  dense: "Dense only",
  bm25: "Keyword only",
  hybrid: "Hybrid (keyword + dense)",
  "kg-local": "LightRAG: local",
  "kg-global": "LightRAG: global",
  "kg-hybrid": "LightRAG: hybrid",
};

function AblationTable({ rows }: { rows: AblationRow[] }) {
  return (
    <div className="table-wrap">
      <table>
        <caption>Retrieval ablation (no LLM calls)</caption>
        <thead>
          <tr>
            <th scope="col">Chunk size</th>
            <th scope="col">Method</th>
            <th scope="col">Rerank</th>
            <th scope="col">Recall@1</th>
            <th scope="col">Recall@3</th>
            <th scope="col">Recall@5</th>
            <th scope="col">MRR@5</th>
            <th scope="col">p50</th>
            <th scope="col">p95</th>
            <th scope="col">Label coverage</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={`${r.chunk_size}-${r.mode}-${r.rerank}-${i}`}>
              <td>{r.chunk_size}</td>
              <td>{MODE_LABELS[r.mode] ?? r.mode}</td>
              <td>{r.rerank ? "Yes" : "No"}</td>
              <td>{pct(r["recall@1"])}</td>
              <td>{pct(r["recall@3"])}</td>
              <td>{pct(r["recall@5"])}</td>
              <td>{r["mrr@5"].toFixed(2)}</td>
              <td>{ms(r.p50_ms)}</td>
              <td>{ms(r.p95_ms)}</td>
              <td>{pct(r.label_coverage)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">
        Label coverage below 100% means the chunker split some answers across chunk boundaries, so no retriever could
        find them at that chunk size.
      </p>
    </div>
  );
}

function AnswerTable({ s }: { s: AnswerSummary }) {
  const rows: [string, string][] = [
    ["Questions evaluated", `${s.questions} (chunk size ${s.chunk_size})`],
    ["Answer accuracy (answerable questions)", pct(s.answer_accuracy_answerable)],
    ["Refusal accuracy (unanswerable questions)", pct(s.refusal_accuracy_unanswerable)],
    ["Citation error rate", pct(s.citation_error_rate)],
    ["Retrieval p50", ms(s.retrieval_p50_ms)],
    ["Model latency p50 / p95", `${ms(s.llm_p50_ms)} / ${ms(s.llm_p95_ms)}`],
    ["Mean tokens per query (in / out)", `${Math.round(s.mean_input_tokens)} / ${Math.round(s.mean_output_tokens)}`],
    ["Cost per 1,000 queries (at the prices you set)", `$${s.cost_per_1000_queries_usd.toFixed(2)}`],
  ];
  return (
    <dl className="metrics">
      {rows.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function EvalResults() {
  if (!ablationRows && !answerSummary) {
    return (
      <div className="notice">
        <strong>Not measured yet.</strong> The results come from the backend evaluation harness, run on a question set
        written against your own documents. Once it runs, copy <code>ablation.json</code> and{" "}
        <code>answer_eval.json</code> from <code>backend/eval/results/</code> into <code>frontend/src/data/</code> and
        rebuild. Results appear here automatically.
      </div>
    );
  }
  return (
    <div className="eval">
      {answerSummary && (
        <>
          <h4>End-to-end answers</h4>
          <AnswerTable s={answerSummary} />
        </>
      )}
      {ablationRows && (
        <>
          <h4>Retrieval ablation</h4>
          <AblationTable rows={ablationRows} />
        </>
      )}
      <p className="muted small">
        These figures come from one question set and one corpus. A small set gives wide error bars, so read them as
        indicators for these documents, not as general performance.
      </p>
    </div>
  );
}
