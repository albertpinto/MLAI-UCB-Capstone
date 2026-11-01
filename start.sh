#!/usr/bin/env bash
# Start Qdrant, the API (:8000) and the UI (:5173). Ctrl-C stops the API and UI (Qdrant keeps running).
set -euo pipefail
cd "$(dirname "$0")"

if ! grep -qE '^GROQ_API_KEY=.+' .env 2>/dev/null; then
  echo "Missing GROQ_API_KEY. Run: cp .env.example .env, then set GROQ_API_KEY in .env" >&2
  exit 1
fi

docker compose up -d qdrant
echo -n "Waiting for Qdrant"
for _ in $(seq 1 30); do
  curl -sf localhost:6335/collections >/dev/null && break
  echo -n "."; sleep 1
done
curl -sf localhost:6335/collections >/dev/null || { echo " Qdrant did not start" >&2; exit 1; }
echo " ok"

[ -d frontend/node_modules ] || (cd frontend && npm install)

trap 'kill 0' EXIT   # stop both child processes when this script exits
uv run uvicorn capstone.main:app --reload &
(cd frontend && npm run dev) &

echo "API: http://localhost:8000/docs   UI: http://localhost:5173"
echo "The first API start downloads the embedding model (a few minutes)."
wait
