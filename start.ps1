# Windows start script: Qdrant (Docker, unless QDRANT_PATH is set in .env), the API (:8000) and the UI (:5173).
# Not tested on Windows by the author: see the User guide in the app if something differs.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# The key for the provider set in .env (LLM_PROVIDER); the check prints what is missing.
uv run python scripts/check_llm_key.py
if ($LASTEXITCODE -ne 0) { exit 1 }

# With QDRANT_PATH set the index is a folder inside the project and Docker is not needed.
if (-not (Select-String -Path .env -Pattern '^QDRANT_PATH=.+' -Quiet)) {
  docker compose up -d qdrant
  Write-Host -NoNewline "Waiting for Qdrant"
  for ($i = 0; $i -lt 30; $i++) {
    try { Invoke-WebRequest -UseBasicParsing http://localhost:6335/collections | Out-Null; break } catch { Write-Host -NoNewline "."; Start-Sleep 1 }
  }
  Write-Host " ok"
}

if (-not (Test-Path frontend/node_modules)) { Push-Location frontend; npm install; Pop-Location }

Start-Process -WindowStyle Minimized -FilePath uv -ArgumentList "run uvicorn capstone.main:app"
Start-Process -WindowStyle Minimized -FilePath npm.cmd -ArgumentList "run dev" -WorkingDirectory "$PSScriptRoot\frontend"
Write-Host "API: http://localhost:8000/docs   UI: http://localhost:5173"
Write-Host "The first API start downloads the embedding model (a few minutes). Stop with .\stop.ps1"
