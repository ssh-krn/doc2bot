import type { OptionsCatalog, PipelineSettings } from "../types";

interface Props {
  catalog: OptionsCatalog;
  value: PipelineSettings;
  onChange: (next: PipelineSettings) => void;
  disabled: boolean;
}

// Techniques that act on the same stage of the pipeline. The backend applies one of each pair.
const OVERLAP_NOTES: Record<string, string> = {
  auto_merging: "Auto-merging takes precedence over parent document retrieval when both are on.",
  contextual_compression: "Compression takes precedence over sentence window when both are on.",
};

const toggle = (list: string[], id: string): string[] =>
  list.includes(id) ? list.filter((x) => x !== id) : [...list, id];

export function PipelinePanel({ catalog, value, onChange, disabled }: Props) {
  const mode = catalog.modes.find((m) => m.id === value.mode);
  const hybridOn = value.techniques.includes("hybrid");
  const searchLabel = hybridOn ? catalog.hybrid_labels[value.mode] : catalog.dense_labels[value.mode];
  const rerankerNames = catalog.rerankers.filter((r) => value.rerankers.includes(r.id)).map((r) => r.label);

  const overlaps = [
    value.techniques.includes("auto_merging") && value.techniques.includes("parent_document") && OVERLAP_NOTES.auto_merging,
    value.techniques.includes("contextual_compression") && value.techniques.includes("sentence_window") &&
      OVERLAP_NOTES.contextual_compression,
  ].filter(Boolean) as string[];

  return (
    <details className="pipeline">
      <summary>
        <span className="pipeline-summary">
          <strong>{mode?.label ?? value.mode}</strong>
          {" · "}
          {searchLabel}
          {" · "}
          {rerankerNames.join(" + ")}
        </span>
      </summary>

      <div className="pipeline-body">
        <fieldset disabled={disabled}>
          <legend>RAG mode</legend>
          <div className="radio-grid">
            {catalog.modes.map((m) => (
              <label key={m.id} className={`option${value.mode === m.id ? " selected" : ""}`}>
                <input
                  type="radio"
                  name="rag-mode"
                  value={m.id}
                  checked={value.mode === m.id}
                  onChange={() => onChange({ ...value, mode: m.id })}
                />
                <span>
                  <span className="option-label">{m.label}</span>
                  <span className="option-hint">{m.description}</span>
                </span>
              </label>
            ))}
          </div>
          {value.mode === "lightrag" && (
            <label className="inline-field">
              LightRAG sub-mode
              <select
                value={value.lightrag_mode}
                onChange={(e) => onChange({ ...value, lightrag_mode: e.target.value })}
              >
                {catalog.lightrag_modes.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label}: {m.description}
                  </option>
                ))}
              </select>
            </label>
          )}
        </fieldset>

        <fieldset disabled={disabled}>
          <legend>Retrieval techniques</legend>
          <div className="check-grid">
            {catalog.techniques.map((t) => {
              const isHybrid = t.id === "hybrid";
              return (
                <label key={t.id} className={`option${value.techniques.includes(t.id) ? " selected" : ""}`}>
                  <input
                    type="checkbox"
                    checked={value.techniques.includes(t.id)}
                    onChange={() => onChange({ ...value, techniques: toggle(value.techniques, t.id) })}
                  />
                  <span>
                    <span className="option-label">{isHybrid ? searchLabel : t.label}</span>
                    <span className="option-hint">{t.description}</span>
                  </span>
                </label>
              );
            })}
          </div>
          {overlaps.map((note) => (
            <p key={note} className="hint">
              {note}
            </p>
          ))}
        </fieldset>

        <fieldset disabled={disabled}>
          <legend>Reranker (choose one or more)</legend>
          <div className="check-grid">
            {catalog.rerankers.map((r) => {
              const checked = value.rerankers.includes(r.id);
              const onlyOne = checked && value.rerankers.length === 1;
              return (
                <label
                  key={r.id}
                  className={`option${checked ? " selected" : ""}${!r.available ? " unavailable" : ""}`}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    disabled={!r.available || onlyOne}
                    onChange={() => onChange({ ...value, rerankers: toggle(value.rerankers, r.id) })}
                  />
                  <span>
                    <span className="option-label">
                      {r.label}
                      {!r.available && " (not configured)"}
                    </span>
                    <span className="option-hint">{r.description}</span>
                  </span>
                </label>
              );
            })}
          </div>
          {value.rerankers.length > 1 && (
            <p className="hint">Several rerankers: their rankings are fused. The refusal threshold uses only the cross-encoder score.</p>
          )}
        </fieldset>
      </div>
    </details>
  );
}
