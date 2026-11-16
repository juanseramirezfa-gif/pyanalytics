# Inicia la app web de pyanalytics (Streamlit)
$ErrorActionPreference = "Stop"
$raiz = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $raiz ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "No se encontro el entorno virtual en .venv" -ForegroundColor Red
    Write-Host "Crealo con:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv"
    Write-Host "  .venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}

Write-Host "Abriendo pyanalytics en http://localhost:8501 ..." -ForegroundColor Green
& $python -m streamlit run (Join-Path $raiz "app\streamlit_app.py") --server.port 8501
