import type { Source } from "../types";
import { sourceAnchorId } from "../lib/anchors";

interface Props {
  text: string;
  sources: Source[];
  messageId: string;
}

const CITATION = /(\[\d+\])/;

/** Renders the answer text, turning [n] markers into links to the matching source. */
export function CitedAnswer({ text, sources, messageId }: Props) {
  const known = new Set(sources.map((s) => s.index));

  return (
    <p className="answer">
      {text.split(CITATION).map((part, i) => {
        const match = /^\[(\d+)\]$/.exec(part);
        if (match && known.has(Number(match[1]))) {
          const n = Number(match[1]);
          return (
            <a key={i} className="cite" href={`#${sourceAnchorId(messageId, n)}`}>
              [{n}]
            </a>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </p>
  );
}
