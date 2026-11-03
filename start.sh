#!/usr/bin/env bash
# Start Qdrant, the API (:8000) and the UI (:5173). Ctrl-C stops the API and UI (Qdrant keeps running).
set -euo pipefail
cd "$(dirname "$0")"

uv run python scripts/check_llm_key.py || exit 1   # the key for the provider set in .env (LLM_PROVIDER)

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
