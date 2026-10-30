import type { ChatMessage } from "../types";
import { CitedAnswer } from "./CitedAnswer";
import { SourceList } from "./SourceList";

export function MessageBubble({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return <div className="bubble user">{message.text}</div>;
  }

  const classes = ["bubble", "assistant", message.status === "error" ? "error" : ""].join(" ").trim();

  return (
    <div className={classes}>
      {message.notes.length > 0 && (
        <details className="notes">
          <summary>What the pipeline did ({message.notes.length})</summary>
          <ul>
            {message.notes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        </details>
      )}
      {message.text ? (
        <CitedAnswer text={message.text} sources={message.sources} messageId={message.id} />
      ) : message.status === "streaming" ? (
        <span className="thinking" role="status" aria-label="Generating answer">
          Thinking…
        </span>
      ) : null}
      {message.facts.length > 0 && (
        <details className="facts">
          <summary>Knowledge graph facts ({message.facts.length})</summary>
          <ul>
            {message.facts.map((f, i) => (
              <li key={i}>{f}</li>
            ))}
          </ul>
        </details>
      )}
      {message.sources.length > 0 && <SourceList messageId={message.id} sources={message.sources} />}
    </div>
  );
}
