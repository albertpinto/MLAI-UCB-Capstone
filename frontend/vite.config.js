import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// API_URL points the dev server at another API (e.g. one on a different collection). /api/* is proxied (not /api-docs, which is a page) to the FastAPI server, so the browser never needs CORS.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, proxy: { "^/api/": process.env.API_URL || "http://localhost:8000" } },
});
