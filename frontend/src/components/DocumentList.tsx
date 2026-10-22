import type { DocumentInfo } from "../types";

interface Props {
  docs: DocumentInfo[];
  loading: boolean;
  onRemove: (docId: string) => Promise<void>;
}

export function DocumentList({ docs, loading, onRemove }: Props) {
  if (loading) return <p className="muted">Loading documents…</p>;
  if (docs.length === 0) return <p className="muted">No documents yet.</p>;

  return (
    <ul className="doc-list">
      {docs.map((d) => (
        <li key={d.doc_id}>
          <div className="doc-meta">
            <strong title={d.filename}>{d.filename}</strong>
            <small>
              {d.pages} pages · {d.chunks} chunks
            </small>
          </div>
          <button
            type="button"
            className="ghost"
            aria-label={`Remove ${d.filename}`}
            onClick={() => void onRemove(d.doc_id)}
          >
            Remove
          </button>
        </li>
      ))}
    </ul>
  );
}
