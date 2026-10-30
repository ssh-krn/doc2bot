import { useEffect, useState } from "react";
import { getOptions } from "../api/client";
import type { OptionsCatalog } from "../types";

/** Loads the modes, techniques and rerankers the backend supports, once. */
export function useOptions() {
  const [catalog, setCatalog] = useState<OptionsCatalog | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getOptions()
      .then((c) => {
        if (!cancelled) setCatalog(c);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "Could not load options");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { catalog, error };
}
