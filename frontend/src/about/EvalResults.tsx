import { ablationRows, answerSummary, type AblationRow, type AnswerSummary } from "../data/evalResults";

const pct = (v: number | null) => (v === null ? "–" : `${(v * 100).toFixed(0)}%`);
const num = (v: number | null, digits = 2) => (v === null ? "–" : v.toFixed(digits));
const ms = (v: number | null) => (v === null ? "–" : `${Math.round(v)} ms`);

const MODE_LABELS: Record<string, string> = {
  dense: "Dense vector search",
  bm25: "Keyword search (BM25)",
  hybrid: "Hybrid (keyword + dense)",
};

const TYPE_LABELS: Record<string, string> = {
  fact: "Fact",
  definition: "Definition",
  comparison: "Comparison",
  numeric: "Numeric",
  multi_hop: "Multi-hop",
};

function AblationTable({ rows }: { rows: AblationRow[] }) {
  return (
    <div className="table-wrap">
      <table>
        <caption>Traditional RAG retrieval, by search method and chunk size (no answer generation)</caption>
        <thead>
          <tr>
            <th scope="col">Chunk size</th>
            <th scope="col">Search</th>
            <th scope="col">Reranked</th>
            <th scope="col">Recall@1</th>
            <th scope="col">Recall@3</th>
            <th scope="col">Recall@5</th>
            <th scope="col">MRR@5</th>
            <th scope="col">p50 latency</th>
            <th scope="col">Gold passages intact</th>
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
              <td>{num(r["mrr@5"])}</td>
              <td>{ms(r.p50_ms)}</td>
              <td>{pct(r.label_coverage)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">
        Gold passages intact: the share of questions whose answer sits inside a single chunk. Below 100% means chunking
        split the answer, so no search method can find it at that size. Dashes are metrics that weren't measured.
      </p>
    </div>
  );
}

function AnswerTable({ s }: { s: AnswerSummary }) {
  const cost =
    s.cost_per_1000_queries_usd === 0 ? "Not priced (set PRICE_IN_PER_M and PRICE_OUT_PER_M)" : `$${s.cost_per_1000_queries_usd.toFixed(2)}`;
  const rows: [string, string][] = [
    ["Questions", `${s.questions} (chunk size ${s.chunk_size})`],
    ["Answer accuracy (answerable)", pct(s.answer_accuracy_answerable)],
    ["Refusal accuracy (unanswerable)", pct(s.refusal_accuracy_unanswerable)],
    ...Object.entries(s.answer_accuracy_by_type ?? {}).map(
      ([t, v]): [string, string] => [`Answer accuracy: ${TYPE_LABELS[t] ?? t}`, pct(v)],
    ),
    ["Citation error rate", pct(s.citation_error_rate)],
    ["Retrieval p50", ms(s.retrieval_p50_ms)],
    ["Model latency p50 / p95", `${ms(s.llm_p50_ms)} / ${ms(s.llm_p95_ms)}`],
    ["Tokens per query (in / out, mean)", `${Math.round(s.mean_input_tokens)} / ${Math.round(s.mean_output_tokens)}`],
    ["Cost per 1,000 queries", cost],
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
        <strong>Not measured yet.</strong> Run the evaluation from <code>backend/</code> with{" "}
        <code>./run_all.sh --answers</code>, then copy <code>eval/results/ablation.json</code> and{" "}
        <code>answer_eval.json</code> into <code>frontend/src/data/</code> and rebuild.
      </div>
    );
  }
  return (
    <div className="eval">
      {answerSummary && (
        <>
          <h4>Traditional RAG: answers</h4>
          <AnswerTable s={answerSummary} />
        </>
      )}
      {ablationRows && (
        <>
          <h4>Traditional RAG: retrieval</h4>
          <AblationTable rows={ablationRows} />
        </>
      )}
      <p className="muted small">
        Traditional RAG is the only mode measured so far. These figures come from one 50-question set on six machine
        learning documents, written for this evaluation. Read them as indicators for this corpus, not general performance.
      </p>
    </div>
  );
}
