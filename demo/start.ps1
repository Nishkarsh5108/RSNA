# Starts the knee MRI workstation on this machine: builds the site once, starts the local server (API + site) on
# http://127.0.0.1:8000, waits for the models to load and opens the browser. Ctrl+C stops it and deletes the cache.
#   powershell -ExecutionPolicy Bypass -File demo\start.ps1          (add -Rebuild after changing the site)
param([switch]$Rebuild)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = if ($env:KNEE_PYTHON) { $env:KNEE_PYTHON } else { "C:\Users\nishk\anaconda3\envs\torch_gpu\python.exe" }
if ($Rebuild -or -not (Test-Path "$here\web\dist\index.html")) {
    if (-not (Test-Path "$here\web\node_modules")) { npm --prefix "$here\web" install }
    npm --prefix "$here\web" run build
    if ($LASTEXITCODE -ne 0) { throw "site build failed" }
}
$env:PYTHONIOENCODING = "utf-8"
$server = Start-Process -FilePath $python -ArgumentList "`"$here\server\app.py`"" -WorkingDirectory "$here\server" -NoNewWindow -PassThru
try {
    $ready = $false
    for ($i = 0; $i -lt 90 -and -not $ready; $i++) {
        Start-Sleep -Milliseconds 700
        try { $ready = (Invoke-RestMethod "http://127.0.0.1:8000/api/status" -TimeoutSec 2).ready } catch { }
    }
    Write-Host ("Workstation " + $(if ($ready) { "ready" } else { "starting (models still loading)" }) + ": http://127.0.0.1:8000")
    Start-Process "http://127.0.0.1:8000"
    Wait-Process -Id $server.Id
} finally {
    if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force -Confirm:$false }
}
