#!/usr/bin/env bash
# Rebuild frontend/public/user-guide.pdf (and its copy Capstone_User_Guide.pdf in the project folder) from the User guide page of the running app (./start.sh first). Run scripts/screenshots.sh before it if the pages changed.
set -euo pipefail
cd "$(dirname "$0")/.."
CHROME=${CHROME:-$(command -v google-chrome || command -v chromium || command -v chromium-browser)}
"$CHROME" --headless=new --no-sandbox --no-pdf-header-footer --virtual-time-budget=25000 --print-to-pdf=frontend/public/user-guide.pdf "${BASE:-http://localhost:5173}/guide" >/dev/null 2>&1
cp frontend/public/user-guide.pdf Capstone_User_Guide.pdf
echo "frontend/public/user-guide.pdf and Capstone_User_Guide.pdf"
