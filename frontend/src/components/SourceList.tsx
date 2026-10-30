import type { Source } from "../types";
import { sourceAnchorId } from "../lib/anchors";
import { MODALITY_LABEL } from "../lib/fileTypes";

interface Props {
  messageId: string;
  sources: Source[];
}

export function SourceList({ messageId, sources }: Props) {
  return (
    <ul className="sources" aria-label="Sources">
      {sources.map((s) => (
        <li key={s.index} id={sourceAnchorId(messageId, s.index)}>
          <details>
            <summary>
              <span className="badge">{s.index}</span>
              {s.filename}, page {s.page}
              {s.modality !== "text" && <span className="modality">{MODALITY_LABEL[s.modality]}</span>}
            </summary>
            {s.modality === "table" ? <pre className="table-src">{s.text}</pre> : <blockquote>{s.text}</blockquote>}
          </details>
        </li>
      ))}
    </ul>
  );
}
