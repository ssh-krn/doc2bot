// Mirrors the backend contract in app/main.py and app/rag.py.

export interface DocumentInfo {
  doc_id: string;
  filename: string;
  chunks: number;
  pages: number;
}

export interface Source {
  index: number; // matches the [n] marker the model writes
  filename: string;
  page: number;
  text: string;
  score: number;
}

// One line of the NDJSON chat stream.
export type ChatEvent =
  | { type: "sources"; data: Source[] }
  | { type: "token"; data: string }
  | { type: "error"; data: string }
  | { type: "done" };

export type MessageStatus = "streaming" | "done" | "error";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  sources: Source[];
  status: MessageStatus;
}
