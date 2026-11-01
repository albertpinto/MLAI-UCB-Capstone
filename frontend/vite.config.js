import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// /api/* is proxied (not /api-docs, which is a page) to the FastAPI server, so the browser never needs CORS.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, proxy: { "^/api/": "http://localhost:8000" } },
});
