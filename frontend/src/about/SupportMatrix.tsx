import { CALLS, MATRIX, MODEL_FIT, SOURCES, type MatrixRow } from "./matrix";

const STRENGTH_CLASS: Record<MatrixRow["strength"], string> = {
  High: "s-high",
  Medium: "s-medium",
  Low: "s-low",
  Contested: "s-contested",
  "None found": "s-none",
};

export function SupportMatrix() {
  return (
    <div className="matrix-block">
      <div className="notice">
        <strong>Not measured here.</strong> Accuracy, latency and cost for these models and modes have not been measured in
        this build, and I couldn't run them. The columns below show what the sources claim, how strong the evidence is,
        and what each model supports and costs. Run the evaluation harness with each setting to fill in measured results.
      </div>

      <h4>Claims, evidence and model fit</h4>
      <div className="table-wrap">
        <table className="matrix">
          <caption>Each row: what the source reports, the check on it, and how each model fits</caption>
          <thead>
            <tr>
              <th scope="col">Feature</th>
              <th scope="col">Published claim</th>
              <th scope="col">Independent check</th>
              <th scope="col">Evidence</th>
              <th scope="col">openai/gpt-oss-120b (Groq)</th>
              <th scope="col">mistral-small-latest (Mistral)</th>
            </tr>
          </thead>
          <tbody>
            {MATRIX.map((r) => (
              <tr key={r.feature}>
                <td>
                  <strong>{r.feature}</strong>
                  <span className="kind-tag">{r.kind}</span>
                  <code className="where-small">{r.where}</code>
                </td>
                <td>{r.claim}</td>
                <td>{r.check}</td>
                <td>
                  <span className={`strength ${STRENGTH_CLASS[r.strength]}`}>{r.strength}</span>
                </td>
                <td>{r.groq}</td>
                <td>{r.mistral}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h4>Model fit: openai/gpt-oss-120b on Groq vs mistral-small-latest</h4>
      <div className="table-wrap">
        <table className="matrix">
          <caption>Capabilities and cost, from provider listings and model cards. Verify prices on the provider sites.</caption>
          <thead>
            <tr>
              <th scope="col">Capability</th>
              <th scope="col">openai/gpt-oss-120b (Groq)</th>
              <th scope="col">mistral-small-latest (Mistral)</th>
            </tr>
          </thead>
          <tbody>
            {MODEL_FIT.map((r) => (
              <tr key={r.capability}>
                <td>
                  <strong>{r.capability}</strong>
                </td>
                <td>{r.groq}</td>
                <td>{r.mistral}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h4>Model calls per question</h4>
      <p className="small muted">
        Counted from the code, assuming the default of five reranked passages. These counts apply to either model, since
        both are called the same way. Multiply by measured tokens and each provider's prices to get cost.
      </p>
      <div className="table-wrap">
        <table className="matrix">
          <caption>Calls per question by mode</caption>
          <thead>
            <tr>
              <th scope="col">Mode</th>
              <th scope="col">Calls</th>
            </tr>
          </thead>
          <tbody>
            {CALLS.map((c) => (
              <tr key={c.mode}>
                <td>
                  <strong>{c.mode}</strong>
                </td>
                <td>{c.calls}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h4>Sources</h4>
      <ul className="sources-list">
        {SOURCES.map((s) => (
          <li key={s.url}>
            <a href={s.url} target="_blank" rel="noreferrer">
              {s.label}
            </a>
          </li>
        ))}
      </ul>
      <p className="small muted">
        Provider figures came from third-party listings and model cards retrieved for this build. They can be out of
        date, so check the providers' own pages before relying on them. Several paper figures (the Self-RAG, LightRAG and
        GraphRAG numbers) came from secondary summaries rather than the papers' tables. Check the papers before quoting them.
      </p>
    </div>
  );
}
