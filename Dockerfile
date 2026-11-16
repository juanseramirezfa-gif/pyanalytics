# Imagen de pyanalytics (Streamlit) para Render / Railway / Fly.io / Docker.
# Uso local:
#   docker build -t pyanalytics .
#   docker run -p 8501:8501 pyanalytics
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# dependencias primero (mejor cache de capas)
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# codigo de la aplicacion
COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
