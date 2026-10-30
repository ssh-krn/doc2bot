import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ command }) => ({
  plugins: [react()],
  // Relative asset paths in the production build, so dist/index.html also works when opened from disk.
  // The dev server keeps root-absolute paths.
  base: command === "build" ? "./" : "/",
  server: {
    proxy: {
      // Forward API calls to the FastAPI backend, so the browser sees one origin.
      "/api": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
}));
