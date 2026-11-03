#!/usr/bin/env bash
# Regenerate the screenshots of the User guide (frontend/public/guide) from the running app: ./start.sh first, with documents ingested.
# Needs Google Chrome or Chromium. The Dashboard shows the last evaluation, the Analysis page the files of eval/grid_search.py and the Quality page the last check, so run those first.
set -euo pipefail
cd "$(dirname "$0")/.."
CHROME=${CHROME:-$(command -v google-chrome || command -v chromium || command -v chromium-browser)}
BASE=${BASE:-http://localhost:5173}
OUT=frontend/public/guide

shot() {  # name, path, height
  "$CHROME" --headless=new --no-sandbox --hide-scrollbars --window-size="1280,$3" --virtual-time-budget=25000 --screenshot="$OUT/$1.png" "$BASE$2" >/dev/null 2>&1
  echo "$1.png"
}

shot ask "/?q=Who%20appoints%20the%20Governor%20of%20a%20State%3F" 660
shot documents "/documents" 800
shot library "/documents?tab=library" 700
shot dashboard "/dashboard" 1900
shot analysis "/analysis" 1500
shot questions "/questions" 1000
shot graph "/graph?entity=governor" 1700
shot quality "/quality" 1500
