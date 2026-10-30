import { useEffect, useRef } from "react";
import type { ChatMessage, OptionsCatalog, PipelineSettings } from "../types";
import { ChatInput } from "./ChatInput";
import { MessageBubble } from "./MessageBubble";
import { PipelinePanel } from "./PipelinePanel";

interface Props {
  messages: ChatMessage[];
  busy: boolean;
  canAsk: boolean;
  catalog: OptionsCatalog | null;
  settings: PipelineSettings;
  onSettings: (next: PipelineSettings) => void;
  onSend: (question: string) => Promise<void>;
  onCancel: () => void;
}

export function ChatView({ messages, busy, canAsk, catalog, settings, onSettings, onSend, onCancel }: Props) {
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
            {canAsk ? "Ask a question about your documents." : "Upload a document to start asking questions."}
          </p>
        )}
        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}
        <div ref={endRef} />
      </div>
      <div className="composer">
        {catalog ? (
          <PipelinePanel catalog={catalog} value={settings} onChange={onSettings} disabled={busy} />
        ) : (
          <p className="muted small">Loading pipeline options…</p>
        )}
        <ChatInput disabled={!canAsk || busy || !catalog} busy={busy} onSend={onSend} onCancel={onCancel} />
      </div>
    </section>
  );
}
