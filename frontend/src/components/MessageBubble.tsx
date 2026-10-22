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
      {message.text ? (
        <CitedAnswer text={message.text} sources={message.sources} messageId={message.id} />
      ) : message.status === "streaming" ? (
        <span className="thinking" role="status" aria-label="Generating answer">
          Thinking…
        </span>
      ) : null}
      {message.sources.length > 0 && <SourceList messageId={message.id} sources={message.sources} />}
    </div>
  );
}
