import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API (FastAPI on 127.0.0.1:8000) does the DICOM parsing and the GPU inference; the dev server only proxies to it.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: false } },
  },
  build: { target: "es2022", chunkSizeWarningLimit: 1500 },
});
