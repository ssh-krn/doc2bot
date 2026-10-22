import { useState, type FormEvent } from "react";

interface Props {
  disabled: boolean;
  busy: boolean;
  onSend: (question: string) => Promise<void>;
  onCancel: () => void;
}

export function ChatInput({ disabled, busy, onSend, onCancel }: Props) {
  const [value, setValue] = useState("");

  function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (disabled || !value.trim()) return;
    void onSend(value);
    setValue("");
  }

  return (
    <form className="input" onSubmit={submit}>
      <input
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="Ask a question about your documents…"
        maxLength={2000}
        disabled={disabled}
        aria-label="Question"
      />
      {busy ? (
        <button type="button" onClick={onCancel}>
          Stop
        </button>
      ) : (
        <button type="submit" disabled={disabled || !value.trim()}>
          Send
        </button>
      )}
    </form>
  );
}
