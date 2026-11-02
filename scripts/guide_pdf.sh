#!/usr/bin/env bash
# Rebuild frontend/public/user-guide.pdf from the User guide page of the running app (./start.sh first). Run scripts/screenshots.sh before it if the pages changed.
set -euo pipefail
cd "$(dirname "$0")/.."
CHROME=${CHROME:-$(command -v google-chrome || command -v chromium || command -v chromium-browser)}
"$CHROME" --headless=new --no-sandbox --no-pdf-header-footer --virtual-time-budget=25000 --print-to-pdf=frontend/public/user-guide.pdf "${BASE:-http://localhost:5173}/guide" >/dev/null 2>&1
echo frontend/public/user-guide.pdf
