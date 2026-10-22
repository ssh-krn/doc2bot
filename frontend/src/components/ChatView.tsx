import { useEffect, useRef } from "react";
import type { ChatMessage } from "../types";
import { ChatInput } from "./ChatInput";
import { MessageBubble } from "./MessageBubble";

interface Props {
  messages: ChatMessage[];
  busy: boolean;
  canAsk: boolean;
  onSend: (question: string) => Promise<void>;
  onCancel: () => void;
}

export function ChatView({ messages, busy, canAsk, onSend, onCancel }: Props) {
  const endRef = useRef<HTMLDivElement>(null);

  // `messages` is intentionally a dependency: the effect should run whenever a message changes.
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
    // oxlint-disable-next-line react/exhaustive-effect-dependencies
  }, [messages]);

  return (
    <section className="chat" aria-label="Chat">
      <div className="messages">
        {messages.length === 0 && (
          <p className="muted empty">
            {canAsk ? "Ask a question about your documents." : "Upload a PDF to start asking questions."}
          </p>
        )}
        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}
        <div ref={endRef} />
      </div>
      <ChatInput disabled={!canAsk || busy} busy={busy} onSend={onSend} onCancel={onCancel} />
    </section>
  );
}
