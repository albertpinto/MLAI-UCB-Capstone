#!/usr/bin/env bash
# Stop the API (:8000) and the UI (:5173) started by start.sh. Qdrant keeps running; pass --all to stop it too (data is kept).
set -euo pipefail
cd "$(dirname "$0")"
root=$(pwd)

# Only processes started from this folder are stopped, so another project's uvicorn or vite is never touched.
stop() {  # $1 = pattern for the command line, $2 = folder the process was started in
  local found=0 pid
  for pid in $(pgrep -f "$1" || true); do
    [ "$(readlink "/proc/$pid/cwd" 2>/dev/null)" = "$2" ] || continue
    kill "$pid" 2>/dev/null && found=1
  done
  [ "$found" = 1 ] && echo "stopped: $1" || echo "not running: $1"
}

stop "uvicorn capstone.main:app" "$root"
stop "vite" "$root/frontend"
stop "npm run dev" "$root/frontend"

if [ "${1:-}" = "--all" ]; then
  docker compose stop qdrant
else
  echo "Qdrant is still running (./stop.sh --all stops it too)."
fi
