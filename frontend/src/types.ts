// Mirrors the backend contract: app/main.py, app/rag.py, app/pipeline/catalog.py.

export type Modality = "text" | "table" | "image" | "equation";

export interface DocumentInfo {
  doc_id: string;
  filename: string;
  kind: string; // file extension without the dot
  chunks: number;
  pages: number;
}

export interface Source {
  index: number; // matches the [n] marker in the answer
  filename: string;
  page: number;
  text: string;
  score: number;
  modality: Modality;
}

// Retrieval options chosen in the pipeline panel.
export interface PipelineSettings {
  mode: string;
  lightrag_mode: string;
  techniques: string[];
  rerankers: string[];
}

export interface CatalogItem {
  id: string;
  label: string;
  description: string;
}

export interface RerankerItem extends CatalogItem {
  available: boolean;
}

export interface OptionsCatalog {
  modes: CatalogItem[];
  lightrag_modes: CatalogItem[];
  techniques: CatalogItem[];
  rerankers: RerankerItem[];
  hybrid_labels: Record<string, string>;
  dense_labels: Record<string, string>;
}

// One line of the NDJSON chat stream.
export type ChatEvent =
  | { type: "sources"; data: Source[] }
  | { type: "note"; data: string }
  | { type: "facts"; data: string[] }
  | { type: "token"; data: string }
  | { type: "error"; data: string }
  | { type: "done" };

export type MessageStatus = "streaming" | "done" | "error";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  sources: Source[];
  facts: string[];
  notes: string[];
  status: MessageStatus;
}

export const DEFAULT_SETTINGS: PipelineSettings = {
  mode: "traditional",
  lightrag_mode: "hybrid",
  techniques: ["hybrid"],
  rerankers: ["cross_encoder"],
};
