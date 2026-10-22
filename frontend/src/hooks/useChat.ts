import { useCallback, useRef, useState } from "react";
import { streamChat } from "../api/client";
import type { ChatMessage } from "../types";

let idCounter = 0;
const newId = () => `msg-${++idCounter}`;

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);

  // Update the assistant message that is currently streaming (always the last one).
  const patchLast = useCallback((patch: (m: ChatMessage) => ChatMessage) => {
    setMessages((prev) => {
      if (prev.length === 0) return prev;
      const next = prev.slice();
      next[next.length - 1] = patch(next[next.length - 1]);
      return next;
    });
  }, []);

  const send = useCallback(
    async (question: string) => {
      const q = question.trim();
      if (!q || busy) return;

      const controller = new AbortController();
      controllerRef.current = controller;
      setBusy(true);
      setMessages((prev) => [
        ...prev,
        { id: newId(), role: "user", text: q, sources: [], status: "done" },
        { id: newId(), role: "assistant", text: "", sources: [], status: "streaming" },
      ]);

      try {
        await streamChat(
          q,
          (event) => {
            switch (event.type) {
              case "sources":
                patchLast((m) => ({ ...m, sources: event.data }));
                break;
              case "token":
                patchLast((m) => ({ ...m, text: m.text + event.data }));
                break;
              case "error":
                patchLast((m) => ({ ...m, text: event.data, status: "error" }));
                break;
              case "done":
                patchLast((m) => (m.status === "streaming" ? { ...m, status: "done" } : m));
                break;
            }
          },
          controller.signal,
        );
      } catch (e) {
        if (controller.signal.aborted) {
          patchLast((m) => ({ ...m, status: "done" }));
        } else {
          const message = e instanceof Error ? e.message : "Request failed";
          patchLast((m) => ({ ...m, text: message, status: "error" }));
        }
      } finally {
        controllerRef.current = null;
        setBusy(false);
      }
    },
    [busy, patchLast],
  );

  const cancel = useCallback(() => controllerRef.current?.abort(), []);

  return { messages, busy, send, cancel };
}
