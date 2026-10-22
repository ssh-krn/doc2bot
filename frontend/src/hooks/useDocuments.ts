import { useCallback, useEffect, useState } from "react";
import { deleteDocument, listDocuments, uploadDocument } from "../api/client";
import type { DocumentInfo } from "../types";

const errorText = (e: unknown, fallback: string) => (e instanceof Error ? e.message : fallback);

export function useDocuments() {
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setDocs(await listDocuments());
      setError(null);
    } catch (e) {
      setError(errorText(e, "Could not load documents"));
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial load. State is set inside promise callbacks, not synchronously in the effect body.
  useEffect(() => {
    let cancelled = false;
    listDocuments()
      .then((d) => {
        if (!cancelled) {
          setDocs(d);
          setError(null);
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(errorText(e, "Could not load documents"));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const upload = useCallback(
    async (file: File) => {
      setUploading(true);
      setError(null);
      try {
        await uploadDocument(file);
        await refresh();
      } catch (e) {
        setError(errorText(e, "Upload failed"));
      } finally {
        setUploading(false);
      }
    },
    [refresh],
  );

  const remove = useCallback(
    async (docId: string) => {
      setError(null);
      try {
        await deleteDocument(docId);
        await refresh();
      } catch (e) {
        setError(errorText(e, "Could not remove document"));
      }
    },
    [refresh],
  );

  return { docs, loading, uploading, error, upload, remove };
}
