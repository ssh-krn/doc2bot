# Doc2Bot frontend

Vite + React + TypeScript. Talks to the FastAPI backend through `/api`, which Vite proxies to `http://localhost:8000`.

```bash
npm install
npm run dev        # http://localhost:5173 (start the backend first)
npm run lint        # oxlint
npm run build      # typecheck + production build
```

## Structure

- `src/api/`: HTTP client. `client.ts` wraps each endpoint. `ndjson.ts` parses the streamed chat response.
- `src/hooks/`: state. `useDocuments` manages the document list and upload/remove. `useChat` turns stream events into messages.
- `src/components/`: presentational components. `CitedAnswer` turns `[n]` markers into links to the source cards.
- `src/lib/`: small shared helpers.
- `src/types.ts`: the API contract, mirroring the backend.

The chat stream is handled by a discriminated union (`ChatEvent`). The `switch` in `useChat` covers every event type, so adding a new event produces a compile error until it's handled.
