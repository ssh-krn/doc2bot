import type { Source } from "../types";
import { sourceAnchorId } from "../lib/anchors";

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
            </summary>
            <blockquote>{s.text}</blockquote>
          </details>
        </li>
      ))}
    </ul>
  );
}
