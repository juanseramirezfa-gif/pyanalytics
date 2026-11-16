# 🐍 pyanalytics

Herramienta en Python para **analizar y explorar datos, generar gráficos, limpiar y normalizar**, con interfaz web y línea de comandos.

Construida a partir del cuaderno *"Analítica con Python"* (21 fuentes sobre Pandas, NumPy, Scikit-learn, series temporales, Dask, Polars y ciclo de vida de ML).

---

## 1. Requisitos e instalación

Se necesita **Python 3.10 o superior**.

```powershell
cd C:\Users\SEBASPC\pyanalytics

# entorno virtual
python -m venv .venv

# dependencias
.venv\Scripts\python.exe -m pip install -r requirements.txt

# (opcional) extras: Dask, Polars, Prophet, Darts, SHAP, LIME, PyMC
.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
```

## 2. Arrancar la app web

```powershell
.\run_app.ps1
# o
.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Se abre en <http://localhost:8501>. Ocho pestañas (con una interfaz propia: tema en `.streamlit/config.toml` y estilos en `app/streamlit_app.py`):

| Pestaña | Qué hace |
|---|---|
| 📁 **Cargar** | CSV, Excel, JSON, Parquet, Pickle, Stata, URL… o un dataset de ejemplo |
| 🔍 **Explorar** | resumen, tipos, nulos, duplicados, estadísticos, correlaciones, atípicos, frecuencias, e **informe interactivo** (skrub) |
| 🧹 **Limpiar** | nombres, espacios, tipos, columnas vacías/constantes, duplicados, imputación, atípicos, codificación |
| ⚖️ **Normalizar** | 14 métodos de escalado + codificación de categóricas, con comparativa antes/después |
| 📊 **Graficar** | histogramas, cajas, dispersión, líneas, barras, circular, correlaciones, pares; descarga PNG |
| 📈 **Series** | remuestreo, medias móviles, ACF/PACF, descomposición, pronóstico con intervalo de confianza |
| 🤖 **Modelar** | comparación de modelos, validación cruzada, métricas, importancia por permutación y **modelo base automático** (skrub) |
| 💾 **Exportar** | descarga en CSV/Excel/Parquet/JSON e informes HTML (propio y exploratorio) |

## 3. Uso por línea de comandos

```powershell
$py = ".venv\Scripts\python.exe"

& $py -m pyanalytics.cli perfil        examples\calidad_agua_sucio.csv
& $py -m pyanalytics.cli limpiar       examples\calidad_agua_sucio.csv -o outputs\limpio.xlsx
& $py -m pyanalytics.cli normalizar    examples\calidad_agua_sucio.csv --metodo minmax -o outputs\norm.csv
& $py -m pyanalytics.cli graficos      examples\calidad_agua_sucio.csv -d outputs\graficos
& $py -m pyanalytics.cli informe       examples\calidad_agua_sucio.csv -o outputs\informe.html
& $py -m pyanalytics.cli convertir     examples\calidad_agua_sucio.csv outputs\calidad_agua.parquet
```

## 4. Uso como librería

```python
import pyanalytics as pa

df = pa.load_data("examples/calidad_agua_sucio.csv")
# ...o directamente un dataset de ejemplo (ingeniería ambiental, Colombia):
# df = pa.cargar_ejemplo("calidad_agua")

# 1) Explorar
perfil = pa.full_profile(df)
print(perfil["overview"])
print(perfil["calidad"]["problemas"])

# 2) Limpiar (con trazabilidad)
limpio, informe = pa.auto_clean(df)
print(informe.summary())

# 3) Normalizar
escalado = pa.normalize(limpio, ["dbo5_mg_l", "solidos_suspendidos_mg_l"], method="robust")

# 4) Graficar
fig = pa.histogram(limpio, "dbo5_mg_l")
pa.save_figure(fig, "outputs/dbo5.png")

# 5) Informe
pa.generate_report(limpio, "outputs/informe.html", title="Calidad del agua")
```

### Escalado sin fuga de información

```python
train_esc, test_esc, escalador = pa.split_scale(train, test, ["salario"], method="zscore")
# el escalador se ajusta SOLO con train y se reutiliza en test
```

### Series temporales

```python
aire = pa.cargar_ejemplo("aire_mensual")            # PM2.5 / PM10 mensual por estación
estacion = aire[aire["estacion"] == "Estacion Centro"]
serie = pa.to_datetime_index(estacion, "fecha", "pm25_ug_m3", freq="MS", agg="mean")["pm25_ug_m3"]
print(pa.describe_series(serie))                     # estacionariedad + estacionalidad
pa.decompose(serie, period=12)                       # tendencia / estacionalidad / residuo
pred = pa.forecast(serie, horizon=12, method="holt_winters")
print(pa.metrics(serie[-12:], pred["pronostico"]))
```

### Modelado

```python
tabla = pa.compare_models(df, target="indice_ica", cv=5)          # regresión
tabla = pa.compare_models(df, target="cumplimiento_norma", cv=5)  # clasificación
res = pa.train_model(df, target="indice_ica", model="random_forest")
print(res["metricas"])
print(pa.permutation_importance(res["pipeline"], res["X_test"], res["y_test"]))
```

## 5. Datasets de ejemplo

Los ejemplos y la app usan datos **sintéticos de ingeniería ambiental en Colombia** (módulo `pyanalytics.datasets`):

| Dataset | Contenido | Columnas destacadas |
|---|---|---|
| `calidad_agua` | monitoreo de ríos, quebradas, humedales y embalses | `ph`, `oxigeno_disuelto_mg_l`, `dbo5_mg_l`, `solidos_suspendidos_mg_l`, `indice_ica`, `cumplimiento_norma` |
| `aire_mensual` | serie mensual de PM2.5 / PM10 por estación | `fecha`, `pm25_ug_m3`, `pm10_ug_m3`, `estacion` |

```python
df = pa.cargar_ejemplo("calidad_agua")     # 815 filas x 17 columnas (con suciedad)
pa.lista_datasets()                        # ['aire_mensual', 'calidad_agua']
```

Los datasets son **intercambiables**: la herramienta funciona con cualquier tabla que cargues.

## 6. Integración con skrub (opcional)

Tras comparar las librerías de perfilado y análisis exploratorio disponibles, se
integró **[skrub](https://skrub-data.org)** (`skrub-data/skrub`, licencia BSD-3,
mantenida por INRIA). Aporta perfilado interactivo y *machine learning with
dataframes* sin salir del ecosistema pandas/scikit-learn.

| Librería | Estado | Compatible con pandas 3 | Elegida |
|---|---|---|---|
| **skrub** | activa (commits diarios) | ✅ | ✅ |
| `fg-data-profiling` (ex ydata-profiling) | activa | ❌ exige `pandas<3.0` | — |
| `sweetviz` | activa | ✅ (pero el target no admite nulos) | alternativa |
| `dataprep` | abandonada (2022) | ❌ | — |

Es una dependencia **opcional**: si no está instalada, la app sigue funcionando
con el perfilador propio. Para habilitarla:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
```

```python
import pyanalytics as pa

df = pa.cargar_ejemplo("calidad_agua")

pa.skrub_disponible()                                   # True/False
pa.informe_eda_html(df, titulo="Calidad del agua")      # HTML interactivo (autocontenido)
pa.guardar_informe_eda(df, "outputs/eda.html")

pa.asociaciones_columnas(df)                            # V de Cramér + Pearson, incluye categóricas
res = pa.modelo_rapido(df, objetivo="Indice ICA")       # línea base en una línea
print(res["metricas"]["r2"])
```

## 7. Estructura del proyecto

```
pyanalytics/
├─ pyanalytics/            paquete (lógica reutilizable)
│  ├─ io.py                carga/guardado: csv, xlsx, json, parquet, pkl, dta, sql, URL
│  ├─ profile.py           perfilado y calidad de datos
│  ├─ clean.py             limpieza con trazabilidad
│  ├─ normalize.py         escalado, codificación, discretización
│  ├─ features.py          ingeniería de características + ColumnTransformer
│  ├─ plots.py             gráficos matplotlib/seaborn y plotly
│  ├─ timeseries.py        series temporales y pronóstico
│  ├─ modeling.py          modelos, validación cruzada, explicabilidad
│  ├─ report.py            informes HTML
│  ├─ eda.py               perfilado interactivo y pipelines (skrub, opcional)
│  ├─ datasets.py          datasets de ejemplo (ingeniería ambiental, Colombia)
│  └─ cli.py               interfaz de terminal
├─ app/streamlit_app.py    interfaz web
├─ .streamlit/config.toml  tema de la app web
├─ examples/               generador de datasets de ejemplo
├─ tests/                  pruebas automáticas
├─ outputs/                resultados (informes, gráficos)
├─ requirements.txt
├─ Dockerfile / Procfile   despliegue alternativo en contenedor
├─ .gitignore
└─ run_app.ps1 / run_app.bat
```

## 8. Métodos de normalización disponibles

| Método | Cuándo usarlo |
|---|---|
| `minmax` | llevar a un rango fijo (0-1) |
| `zscore` | distribución aproximadamente normal |
| `robust` | hay valores atípicos (usa mediana e IQR) |
| `maxabs` | datos dispersos, conserva los ceros |
| `decimal` | escalado por orden de magnitud |
| `log` / `log1p` | distribuciones sesgadas a la derecha |
| `sqrt` | conteos, varianza ~ media |
| `boxcox` | normalizar (requiere valores > 0) |
| `yeojohnson` | como Box-Cox pero admite negativos |
| `quantile` | forzar uniforme/normal, muy robusto |
| `rank` | eliminar el efecto de los valores extremos |
| `l2` | normalizar filas (texto, embeddings) |
| `binary` | umbralizar a 0/1 |

## 9. Pruebas

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

## 10. Notas

- Todos los gráficos estáticos usan el backend `Agg`, así que funcionan sin pantalla.
- `generate_report` produce un HTML autocontenido (imágenes incrustadas en base64).
- Las dependencias opcionales (`dask`, `polars`, `prophet`, `darts`, `shap`, `lime`, `pymc`, `skrub`) no son necesarias para usar la herramienta.

## 11. Despliegue en la nube

La forma más sencilla de publicar la app es **Streamlit Community Cloud**
(gratis) conectada a un repositorio de **GitHub**.

> ⚠️ **Vercel no puede ejecutar Streamlit.** Vercel es *serverless*: no mantiene
> el servidor Python persistente ni los WebSockets que Streamlit necesita. Para
> una app de Streamlit hay que usar un host con proceso de larga duración.

### 11.1 Subir el proyecto a GitHub

```powershell
# Git (una sola vez):  winget install Git.Git -e
git init
git add .
git commit -m "pyanalytics: app de analitica con Streamlit"
git branch -M main

# crea un repositorio vacio en https://github.com/new y despues:
git remote add origin https://github.com/TU_USUARIO/pyanalytics.git
git push -u origin main
```

### 11.2 Desplegar en Streamlit Community Cloud

1. Entra en <https://share.streamlit.io> e inicia sesión con GitHub.
2. **New app → Deploy a public app from GitHub**.
3. Rellena:
   - **Repository**: `TU_USUARIO/pyanalytics`
   - **Branch**: `main`
   - **Main file path**: `app/streamlit_app.py`
4. Pulsa **Deploy**. La primera vez instala `requirements.txt` (tarda unos minutos).
5. Obtendrás una URL pública tipo `https://pyanalytics.streamlit.app`.

Notas:

- La app **funciona sin las dependencias opcionales**; el informe interactivo de
  skrub solo aparece si están instaladas. Para tenerlo en la nube añade
  `skrub>=0.11` a `requirements.txt`.
- No subas al repositorio datos reales o sensibles: los datasets de ejemplo son
  sintéticos.

### 11.3 Alternativas (contenedor)

| Plataforma | ¿Ejecuta Streamlit? | Notas |
|---|---|---|
| Streamlit Community Cloud | ✅ | gratis, desde GitHub, la más simple |
| Render / Railway / Fly.io | ✅ | usan `Dockerfile` o `Procfile` |
| Hugging Face Spaces | ✅ | SDK "Streamlit" |
| Vercel | ❌ | serverless: no soporta Streamlit |

Incluye un `Dockerfile` y un `Procfile` listos para los hosts de contenedor:

```powershell
docker build -t pyanalytics .
docker run -p 8501:8501 pyanalytics
```
