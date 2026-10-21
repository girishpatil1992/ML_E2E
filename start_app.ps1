$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $root "venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    $venvPython = (Get-Command python -ErrorAction Stop).Source
}

$env:API_BASE_URL = "http://127.0.0.1:8000"

Write-Host "Starting FastAPI on http://127.0.0.1:8000"
Start-Process -FilePath $venvPython -ArgumentList "-m uvicorn api:app --host 0.0.0.0 --port 8000" -WorkingDirectory $root

Start-Sleep -Seconds 2

Write-Host "Starting Streamlit on http://127.0.0.1:8501"
Start-Process -FilePath $venvPython -ArgumentList "-m streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501" -WorkingDirectory $root

Write-Host ""
Write-Host "Open: http://127.0.0.1:8501"
Write-Host "FastAPI health check: http://127.0.0.1:8000/health"
Write-Host ""
Write-Host "Both services are running in the background."
