@echo off
REM Inicia la app web de pyanalytics (Streamlit)
setlocal
set RAIZ=%~dp0
if not exist "%RAIZ%.venv\Scripts\python.exe" (
  echo No se encontro el entorno virtual en .venv
  echo Creelo con:
  echo   python -m venv .venv
  echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
  exit /b 1
)
echo Abriendo pyanalytics en http://localhost:8501 ...
"%RAIZ%.venv\Scripts\python.exe" -m streamlit run "%RAIZ%app\streamlit_app.py" --server.port 8501
endlocal
