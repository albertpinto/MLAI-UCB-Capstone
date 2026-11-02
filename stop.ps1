# Windows stop script: stops whatever listens on the API (:8000) and UI (:5173) ports. Qdrant keeps running; -All stops it too.
param([switch]$All)
Set-Location $PSScriptRoot
foreach ($port in 8000, 5173) {
  $owners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
  if ($owners) { $owners | ForEach-Object { Stop-Process -Id $_ -Force }; Write-Host "stopped port $port" } else { Write-Host "nothing on port $port" }
}
if ($All) { docker compose stop qdrant } else { Write-Host "Qdrant is still running (.\stop.ps1 -All stops it too)." }
