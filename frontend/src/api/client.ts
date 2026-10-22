import type { ChatEvent, DocumentInfo } from "../types";
import { readNdjson } from "./ndjson";

const BASE = "/api";

async function errorMessage(res: Response): Promise<string> {
  try {
    const body: unknown = await res.json();
    if (body && typeof body === "object" && "detail" in body && typeof body.detail === "string") {
      return body.detail;
    }
  } catch {
    // Body wasn't JSON; fall through to the status text.
  }
  return res.statusText || `Request failed (HTTP ${res.status})`;
}

export async function listDocuments(signal?: AbortSignal): Promise<DocumentInfo[]> {
  const res = await fetch(`${BASE}/documents`, { signal });
  if (!res.ok) throw new Error(await errorMessage(res));
  return (await res.json()) as DocumentInfo[];
}

export async function uploadDocument(file: File): Promise<DocumentInfo> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/documents`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await errorMessage(res));
  return (await res.json()) as DocumentInfo;
}

export async function deleteDocument(docId: string): Promise<void> {
  const res = await fetch(`${BASE}/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await errorMessage(res));
}

export async function streamChat(
  question: string,
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!res.ok) throw new Error(await errorMessage(res));
  if (!res.body) throw new Error("Response has no body");
  await readNdjson<ChatEvent>(res.body, onEvent);
}
