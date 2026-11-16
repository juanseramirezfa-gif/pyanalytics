"""Interfaz web de pyanalytics (Streamlit).

Ejecutar:
    streamlit run app/streamlit_app.py

Permite cargar un archivo, explorarlo, limpiarlo, normalizarlo, graficarlo,
analizar series temporales y entrenar modelos, todo sin escribir codigo.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# permite ejecutar el archivo directamente desde la carpeta app/
RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import pyanalytics as pa  # noqa: E402

st.set_page_config(
    page_title="pyanalytics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)
pa.set_style()

EXTENSIONES = ["csv", "txt", "tsv", "xlsx", "xlsm", "json", "jsonl", "parquet", "feather", "pkl", "dta", "ods"]

# --------------------------------------------------------------------------- #
# estilos de la interfaz
# --------------------------------------------------------------------------- #
_ESTILOS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp, button, input, textarea, select {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

.stApp {
    background:
        radial-gradient(1100px 520px at 12% -12%, #E4F7EF 0%, rgba(228,247,239,0) 58%),
        radial-gradient(900px 480px at 102% -6%, #E8F2FF 0%, rgba(232,242,255,0) 55%),
        #F5F9F7;
}

/* --- cromo de Streamlit --- */
#MainMenu, footer, [data-testid="stStatusWidget"] { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"] { background: linear-gradient(90deg, #0E9F6E, #34D399); }
[data-testid="stAppViewBlockContainer"], .block-container {
    padding-top: 1.4rem; padding-bottom: 3.5rem; max-width: 1340px;
}

/* --- tipografia --- */
h1, h2, h3, h4 { color: #0F172A !important; letter-spacing: -0.02em; }
h2 {
    font-weight: 800 !important; font-size: 1.45rem;
    border-left: 5px solid #0E9F6E; padding-left: 13px; border-radius: 3px;
}
h3 { font-weight: 700 !important; }

/* --- hero --- */
.hero {
    position: relative; overflow: hidden; border-radius: 24px;
    padding: 30px 34px; margin-bottom: 4px;
    background: linear-gradient(120deg, #06372B 0%, #0E9F6E 58%, #34D399 100%);
    box-shadow: 0 26px 60px -32px rgba(6, 95, 70, 0.95);
}
.hero::after {
    content: ""; position: absolute; inset: 0;
    background: radial-gradient(520px 240px at 90% -40%, rgba(255, 255, 255, 0.32), transparent 62%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block; padding: 5px 13px; border-radius: 999px; margin-bottom: 13px;
    font-size: 0.74rem; font-weight: 700; letter-spacing: 0.02em;
    background: rgba(255, 255, 255, 0.18); color: #EAFBF4;
}
.hero h1 { color: #FFFFFF !important; font-size: 1.95rem; font-weight: 800; margin: 0 0 7px 0; }
.hero p { color: #D6F5E9 !important; margin: 0; font-size: 1.02rem; max-width: 78ch; }

/* --- tarjetas de metrica --- */
[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #E5EEEA; border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04), 0 12px 30px -20px rgba(6, 95, 70, 0.45);
    transition: transform 0.16s ease, box-shadow 0.16s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 36px -18px rgba(6, 95, 70, 0.55);
}
[data-testid="stMetricLabel"] p { color: #64748B !important; font-weight: 600; font-size: 0.82rem; }
[data-testid="stMetricValue"] { color: #065F46; font-weight: 800; }

/* --- pestanas --- */
.stTabs [role="tablist"] { gap: 6px; border-bottom: 1px solid #E4EDE9; }
.stTabs [data-testid="stTab"] {
    height: 44px; padding: 0 16px; border-radius: 12px 12px 0 0;
    color: #475569; background: transparent; transition: background 0.15s ease;
}
.stTabs [data-testid="stTab"] p { font-weight: 700; }
.stTabs [data-testid="stTab"]:hover { background: #EFF7F3; }
.stTabs [data-testid="stTab"]:hover p { color: #065F46; }
.stTabs [data-testid="stTab"][aria-selected="true"] { background: #E7F8F1 !important; }
.stTabs [data-testid="stTab"][aria-selected="true"] p { color: #065F46 !important; }
.react-aria-SelectionIndicator { display: none !important; }
div[data-baseweb="tab-panel"], [role="tabpanel"] { padding-top: 18px; }

/* --- botones --- */
.stButton > button, .stDownloadButton > button {
    border-radius: 12px; border: 1px solid #D8E5DF; font-weight: 700;
    transition: transform 0.15s ease, filter 0.15s ease, background 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: #0E9F6E; color: #065F46; }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
    background: linear-gradient(120deg, #0E9F6E, #10B981); border: none; color: #FFFFFF;
    box-shadow: 0 10px 24px -12px rgba(16, 185, 129, 0.85);
}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {
    color: #FFFFFF; filter: brightness(1.06); transform: translateY(-1px);
}

/* --- contenedores --- */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid #E5EEEA; border-radius: 14px; overflow: hidden; background: #FFFFFF;
}
[data-testid="stExpander"] {
    border: 1px solid #E5EEEA; border-radius: 14px; background: #FFFFFF; overflow: hidden;
}
[data-testid="stAlert"] { border-radius: 13px; }
hr { border-color: #E4EDE9; }
.stTextInput input, .stNumberInput input, .stTextArea textarea,
[data-baseweb="select"] > div, [data-baseweb="input"] { border-radius: 12px !important; }
[data-testid="stFileUploaderDropzone"] { border-radius: 14px; border: 1.5px dashed #B9D9CD; background: #FBFEFC; }

/* --- barra lateral --- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #052E23 0%, #083D2E 58%, #0A4A38 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #E8F6F0 !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255, 255, 255, 0.12); }
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255, 255, 255, 0.10); color: #EAFBF4; border: 1px solid rgba(255, 255, 255, 0.20);
}
[data-testid="stSidebar"] .stButton > button:hover { background: rgba(255, 255, 255, 0.20); color: #FFFFFF; }

.marca { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; }
.marca-logo {
    width: 46px; height: 46px; border-radius: 13px; display: grid; place-items: center; font-size: 23px;
    background: linear-gradient(135deg, #34D399, #0E9F6E);
    box-shadow: 0 12px 24px -12px rgba(16, 185, 129, 0.95);
}
.marca-titulo { font-weight: 800; font-size: 1.18rem; color: #FFFFFF; letter-spacing: -0.01em; }
.marca-sub { font-size: 0.72rem; color: #9FD9C6; }

.pasos { display: flex; flex-direction: column; gap: 6px; margin-top: 10px; }
.paso {
    display: flex; align-items: center; gap: 10px; padding: 8px 11px; border-radius: 11px;
    background: rgba(255, 255, 255, 0.05); color: #DCF3EA; font-size: 0.86rem; font-weight: 600;
    border: 1px solid rgba(255, 255, 255, 0.06);
}
.paso span {
    width: 22px; height: 22px; border-radius: 50%; display: grid; place-items: center;
    background: rgba(16, 185, 129, 0.92); color: #04231A; font-size: 0.72rem; font-weight: 800;
}

.chip {
    margin-top: 12px; padding: 9px 12px; border-radius: 11px; text-align: center;
    background: rgba(16, 185, 129, 0.16); color: #BFF3E2; font-size: 0.8rem; font-weight: 700;
    border: 1px solid rgba(16, 185, 129, 0.32);
}
.chip-off { background: rgba(255, 255, 255, 0.06); color: #9FB8AF; border-color: rgba(255, 255, 255, 0.10); }

.pie { margin-top: 26px; text-align: center; color: #94A3B8; font-size: 0.8rem; }
</style>
"""

st.markdown(_ESTILOS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# carga de datos
# --------------------------------------------------------------------------- #
def _leer_csv(datos: bytes, **kwargs) -> pd.DataFrame:
    """Lee un CSV probando varias codificaciones (utf-8, cp1252, latin-1)."""
    ultimo_error: Exception | None = None
    for codificacion in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(datos), encoding=codificacion, **kwargs)
        except UnicodeDecodeError as exc:
            ultimo_error = exc
    raise ultimo_error if ultimo_error is not None else ValueError("No se pudo leer el archivo CSV.")


def _leer_subida(archivo) -> pd.DataFrame:
    nombre = archivo.name.lower()
    datos = archivo.getvalue()
    if nombre.endswith(".csv") or nombre.endswith(".txt"):
        try:
            return _leer_csv(datos, sep=None, engine="python")
        except Exception:
            return _leer_csv(datos)
    if nombre.endswith(".tsv"):
        return _leer_csv(datos, sep="\t")
    return pa.load_data(io.BytesIO(datos), kind=Path(nombre).suffix.lstrip("."))


def _datos_ejemplo(nombre: str) -> pd.DataFrame:
    """Devuelve uno de los datasets de ejemplo (ingenieria ambiental, Colombia)."""
    return pa.cargar_ejemplo(nombre)


def barra_lateral() -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="marca">
              <div class="marca-logo">🌿</div>
              <div>
                <div class="marca-titulo">pyanalytics</div>
                <div class="marca-sub">v{pa.__version__} · analitica en Python</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="pasos">
              <div class="paso"><span>1</span> Cargar datos</div>
              <div class="paso"><span>2</span> Explorar</div>
              <div class="paso"><span>3</span> Limpiar</div>
              <div class="paso"><span>4</span> Normalizar</div>
              <div class="paso"><span>5</span> Graficar</div>
              <div class="paso"><span>6</span> Series y modelos</div>
              <div class="paso"><span>7</span> Exportar informe</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        df = _df_actual()
        if df is not None:
            st.markdown(
                f'<div class="chip">✅ {df.shape[0]:,} filas · {df.shape[1]} columnas</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="chip chip-off">Sin datos cargados</div>', unsafe_allow_html=True)
        if st.button("🔄 Reiniciar datos", width="stretch", key="btn_reiniciar"):
            for clave in list(st.session_state.keys()):
                if clave.startswith("df"):
                    del st.session_state[clave]
            st.rerun()


# --------------------------------------------------------------------------- #
# pestanas
# --------------------------------------------------------------------------- #
def tab_carga() -> None:
    st.header("1. Cargar datos")
    col1, col2 = st.columns([2, 1])

    with col1:
        archivo = st.file_uploader("Sube un archivo", type=EXTENSIONES, key="upload_archivo")
        if archivo is not None:
            try:
                df = _leer_subida(archivo)
                st.session_state["df_original"] = df
                st.success(f"Cargado {archivo.name}: {df.shape[0]:,} filas x {df.shape[1]} columnas")
            except Exception as exc:
                st.error(f"No se pudo leer el archivo: {exc}")

        with st.expander("…o carga desde una URL"):
            url = st.text_input("URL directa al archivo", key="url_archivo")
            if st.button("Cargar URL", key="btn_cargar_url") and url:
                try:
                    st.session_state["df_original"] = pa.load_data(url)
                    st.success("Datos cargados desde la URL.")
                except Exception as exc:
                    st.error(f"Error al cargar la URL: {exc}")

    with col2:
        st.markdown("**Datos de ejemplo**")
        ejemplo = st.selectbox(
            "Dataset",
            pa.lista_datasets(),
            format_func=lambda n: {
                "calidad_agua": "Calidad del agua (rios y quebradas)",
                "aire_mensual": "Calidad del aire (PM2.5 / PM10 mensual)",
            }.get(n, n),
            label_visibility="collapsed",
            key="sel_dataset",
        )
        if st.button("Generar dataset de ejemplo", width="stretch", key="btn_generar_ejemplo"):
            st.session_state["df_original"] = _datos_ejemplo(ejemplo)
            st.success(f"Dataset de ejemplo generado: {ejemplo}.")
        st.caption(
            "Datos sinteticos de ingenieria ambiental en Colombia, con nulos, "
            "duplicados, texto sucio y atipicos a proposito."
        )

    if "df_original" in st.session_state:
        st.divider()
        df = st.session_state["df_original"]
        st.subheader("Vista previa")
        st.dataframe(df.head(100), width="stretch", height=320)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Filas", f"{df.shape[0]:,}")
        c2.metric("Columnas", df.shape[1])
        c3.metric("Nulos", f"{df.isna().sum().sum():,}")
        c4.metric("Duplicados", f"{df.duplicated().sum():,}")

        if "df_limpio" not in st.session_state:
            if st.button("Usar tal cual (sin limpiar)", type="primary", key="btn_usar_tal_cual"):
                st.session_state["df_limpio"] = df.copy()
                st.session_state["informe_limpieza"] = "Sin limpieza aplicada."
                st.rerun()
    else:
        st.info("Sube un archivo o genera el dataset de ejemplo para empezar.")


def _df_actual() -> pd.DataFrame | None:
    if "df_limpio" in st.session_state:
        return st.session_state["df_limpio"]
    return st.session_state.get("df_original")


def tab_explorar() -> None:
    st.header("2. Explorar")
    df = _df_actual()
    if df is None:
        st.info("Primero carga datos en la pestana **Cargar**.")
        return

    perfil = pa.full_profile(df)
    ov, calidad = perfil["overview"], perfil["calidad"]

    c = st.columns(6)
    c[0].metric("Filas", f"{ov['filas']:,}")
    c[1].metric("Columnas", ov["columnas"])
    c[2].metric("Memoria", ov["memoria_legible"])
    c[3].metric("% nulos", f"{ov['porcentaje_nulos']}%")
    c[4].metric("Duplicados", f"{ov['filas_duplicadas']:,}")
    c[5].metric("Calidad", f"{calidad['puntaje']}/100")

    if calidad["problemas"]:
        st.warning("**Problemas detectados:**\n\n" + "\n".join(f"- {p}" for p in calidad["problemas"]))
    else:
        st.success("No se detectaron problemas relevantes de calidad.")

    sub = st.tabs(["Columnas", "Nulos", "Duplicados", "Numericas", "Categoricas", "Correlaciones", "Atipicos", "Frecuencias", "Informe interactivo"])

    with sub[0]:
        st.dataframe(perfil["columnas"], width="stretch", height=420)
    with sub[1]:
        st.dataframe(perfil["nulos"], width="stretch", height=380)
        if df.isna().any().any():
            st.pyplot(pa.missing_matrix(df))
    with sub[2]:
        rep = perfil["duplicados"]
        st.metric("Filas duplicadas", f"{rep['n_filas_duplicadas_exactas']:,}")
        if rep["n_filas_duplicadas_exactas"]:
            st.dataframe(rep["muestra"], width="stretch")
    with sub[3]:
        if perfil["estadisticos_numericos"].empty:
            st.info("No hay columnas numericas.")
        else:
            st.dataframe(perfil["estadisticos_numericos"].style.format("{:.3f}"), width="stretch", height=420)
    with sub[4]:
        st.dataframe(perfil["estadisticos_categoricos"], width="stretch", height=380)
    with sub[5]:
        if perfil["correlaciones"].empty:
            st.info("Se necesitan al menos dos columnas numericas.")
        else:
            metodo = st.radio("Metodo", ["pearson", "spearman", "kendall"], horizontal=True, key="radio_metodo_corr")
            st.plotly_chart(pa.plots.interactive_corr(df, method=metodo), width="stretch")
            st.markdown("**Pares muy correlacionados (|r| ≥ 0.8)**")
            st.dataframe(perfil["correlaciones_altas"], width="stretch")
    with sub[6]:
        st.dataframe(perfil["atipicos"], width="stretch", height=380)
    with sub[7]:
        col = st.selectbox("Columna", list(df.columns), key="sel_frec_columna")
        top = st.slider("Valores a mostrar", 5, 50, 15, key="slider_top")
        st.dataframe(pa.value_counts_table(df, col, top=top), width="stretch")
    with sub[8]:
        st.markdown(
            "Perfilado **interactivo** con [skrub](https://skrub-data.org) "
            "(libreria opcional): distribuciones, valores representativos, nulos y asociaciones."
        )
        if not pa.skrub_disponible():
            st.info(
                "skrub no esta instalado: se usara el generador de informes propio. "
                "Para el informe interactivo ejecuta `pip install -r requirements-optional.txt`."
            )
        titulo_eda = st.text_input("Titulo del informe", "Perfil exploratorio", key="titulo_eda")
        if st.button("Generar informe exploratorio", type="primary", key="btn_eda_tab2"):
            with st.spinner("Perfilando los datos..."):
                try:
                    st.session_state["html_eda"] = pa.informe_eda_html(df, titulo=titulo_eda)
                    st.success("Informe exploratorio generado.")
                except Exception as exc:
                    st.error(f"No se pudo generar el informe: {exc}")
        if st.session_state.get("html_eda"):
            st.download_button(
                "Descargar informe HTML",
                st.session_state["html_eda"].encode("utf-8"),
                file_name="perfil_exploratorio.html",
                mime="text/html",
                width="stretch",
                key="dl_eda_tab2",
            )
            st.iframe(st.session_state["html_eda"], height=720)


def tab_limpiar() -> None:
    st.header("3. Limpiar")
    df = _df_actual()
    if df is None:
        st.info("Primero carga datos.")
        return

    st.caption("Aplica los pasos que necesites y pulsa **Aplicar limpieza**.")
    col1, col2 = st.columns(2)
    with col1:
        nombres = st.checkbox("Normalizar nombres de columna (snake_case)", value=True, key="chk_nombres")
        espacios = st.checkbox("Recortar espacios en texto", value=True, key="chk_espacios")
        tipos = st.checkbox("Detectar tipos automaticamente (numeros/fechas como texto)", value=True, key="chk_tipos")
        constantes = st.checkbox("Eliminar columnas constantes", value=True, key="chk_constantes")
        vacias = st.checkbox("Eliminar columnas muy vacias", value=True, key="chk_vacias")
        umbral = st.slider("Umbral de nulos para eliminar columna", 0.3, 1.0, 0.6, 0.05, disabled=not vacias, key="slider_umbral")
    with col2:
        duplicados = st.checkbox("Eliminar duplicados", value=True, key="chk_duplicados")
        imputar = st.checkbox("Rellenar nulos automaticamente", value=True, key="chk_imputar")
        estrategia = st.selectbox(
            "Estrategia de imputacion",
            ["auto", "mean", "median", "mode", "ffill", "interpolate", "zero", "drop"],
            disabled=not imputar,
            key="sel_estrategia",
        )
        atipicos = st.checkbox("Recortar atipicos (IQR)", value=False, key="chk_atipicos")
        eliminar_atipicos = st.checkbox("Eliminar filas con atipicos", value=False, key="chk_eliminar_atipicos")
        codificar = st.selectbox("Codificar categoricas", ["no", "onehot", "ordinal", "frequency"], index=0, key="sel_codificar")

    if st.button("Aplicar limpieza", type="primary", key="btn_aplicar_limpieza"):
        pasos: list[dict] = []
        if nombres:
            pasos.append({"funcion": "normalize_column_names"})
        if espacios:
            pasos.append({"funcion": "strip_whitespace"})
        if tipos:
            pasos.append({"funcion": "convert_types", "auto": True})
        if vacias:
            pasos.append({"funcion": "drop_empty_columns", "threshold": umbral})
        if constantes:
            pasos.append({"funcion": "drop_constant_columns"})
        if duplicados:
            pasos.append({"funcion": "drop_duplicates"})
        if imputar:
            pasos.append({"funcion": "impute_missing", "strategy": estrategia})
        if atipicos:
            pasos.append({"funcion": "cap_outliers", "method": "iqr"})
        if eliminar_atipicos:
            pasos.append({"funcion": "remove_outliers", "method": "iqr"})

        limpio, informe = pa.clean_pipeline(df, pasos)
        if codificar != "no":
            limpio = pa.encode_categorical(limpio, method=codificar)
        st.session_state["df_limpio"] = limpio
        st.session_state["informe_limpieza"] = informe.summary()
        st.session_state.pop("df_antes_escalado", None)
        st.session_state.pop("df_columnas_escaladas", None)
        st.success(f"Listo: {limpio.shape[0]:,} filas x {limpio.shape[1]} columnas")
        st.rerun()

    if "informe_limpieza" in st.session_state:
        st.divider()
        st.subheader("Trazabilidad de la limpieza")
        st.code(st.session_state["informe_limpieza"], language="text")
        if st.button("Volver a los datos originales", key="btn_volver_originales"):
            st.session_state.pop("df_limpio", None)
            st.session_state.pop("df_antes_escalado", None)
            st.session_state.pop("df_columnas_escaladas", None)
            st.rerun()


def tab_normalizar() -> None:
    st.header("4. Normalizar y transformar")
    df = _df_actual()
    if df is None:
        st.info("Primero carga datos.")
        return

    numericas = pa.numeric_columns(df)
    if not numericas:
        st.warning("No hay columnas numericas para escalar.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        columnas = st.multiselect("Columnas", numericas, default=numericas[: min(4, len(numericas))], key="ms_columnas_escalar")
    with col2:
        metodo = st.selectbox("Metodo de escalado", list(pa.SCALE_METHODS), index=list(pa.SCALE_METHODS).index("zscore"), key="sel_metodo_escalado")
    with col3:
        reemplazar = st.radio("Modo", ["Anadir nuevas columnas", "Reemplazar originales"], index=0, key="radio_modo_escalado")

    if columnas:
        sugerencias = {c: pa.recommend_method(df[c]) for c in columnas}
        st.caption("Sugerencia automatica por columna: " + ", ".join(f"{c} → {m}" for c, m in sugerencias.items()))

    if st.button("Aplicar escalado", type="primary", key="btn_aplicar_escalado") and columnas:
        st.session_state["df_antes_escalado"] = df.copy()
        st.session_state["df_columnas_escaladas"] = list(columnas)
        escalado = pa.normalize(df, columnas, method=metodo, keep_original=(reemplazar.startswith("Anadir")))
        st.session_state["df_limpio"] = escalado
        st.success("Escalado aplicado. Revisa los resultados abajo.")
        st.rerun()

    st.divider()
    st.subheader("Comparativa antes / despues")
    base = st.session_state.get("df_antes_escalado")
    actual = _df_actual()
    escaladas = st.session_state.get("df_columnas_escaladas") or []
    if base is None or actual is None or not escaladas:
        st.info("Aplica un escalado para ver la comparativa antes / despues.")
        return

    nuevas = [c for c in actual.columns if c not in base.columns] or escaladas
    antes = pa.numeric_stats(base[escaladas]) if all(c in base.columns for c in escaladas) else pd.DataFrame()
    despues = pa.numeric_stats(actual[nuevas]) if nuevas else pd.DataFrame()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Antes**")
        st.dataframe(antes.style.format("{:.3f}") if not antes.empty else antes, width="stretch")
    with c2:
        st.markdown("**Despues**")
        st.dataframe(despues.style.format("{:.3f}") if not despues.empty else despues, width="stretch")

    st.markdown("**Distribucion antes vs despues**")
    comparar = pd.concat(
        [
            pd.DataFrame({"columna": escaladas, "valor": [base[c].mean() for c in escaladas], "tipo": "original"}),
            pd.DataFrame({"columna": escaladas, "valor": [actual[c].mean() for c in escaladas if c in actual.columns], "tipo": "escalado"}),
        ]
    )
    import plotly.express as px

    st.plotly_chart(px.bar(comparar, x="columna", y="valor", color="tipo", barmode="group"), width="stretch")


def tab_graficar() -> None:
    st.header("5. Graficar")
    df = _df_actual()
    if df is None:
        st.info("Primero carga datos.")
        return

    numericas = pa.numeric_columns(df)
    categoricas = pa.categorical_columns(df)
    fechas = pa.datetime_columns(df)
    tipo = st.selectbox(
        "Tipo de grafico",
        ["Histograma", "Caja", "Dispersion", "Lineas", "Barras", "Frecuencia", "Circular", "Correlaciones", "Pares"],
        key="sel_tipo_grafico",
    )

    try:
        if tipo == "Histograma" and numericas:
            col = st.selectbox("Columna", numericas, key="sel_hist_col")
            color = st.selectbox("Colorear por (opcional)", ["ninguno"] + categoricas, key="sel_hist_color")
            st.plotly_chart(pa.plots.interactive_histogram(df, col, color=None if color == "ninguno" else color), width="stretch")
        elif tipo == "Caja" and numericas:
            col = st.selectbox("Columna", numericas, key="sel_caja_col")
            by = st.selectbox("Agrupar por (opcional)", ["ninguno"] + categoricas, key="sel_caja_by")
            st.plotly_chart(pa.plots.interactive_box(df, col, by=None if by == "ninguno" else by), width="stretch")
        elif tipo == "Dispersion" and len(numericas) >= 2:
            x = st.selectbox("Eje X", numericas, key="sel_disp_x")
            y = st.selectbox("Eje Y", numericas, index=1, key="sel_disp_y")
            color = st.selectbox("Color (opcional)", ["ninguno"] + categoricas, key="sel_disp_color")
            tendencia = st.checkbox("Linea de tendencia", value=True, key="chk_disp_tendencia")
            st.plotly_chart(pa.plots.interactive_scatter(df, x, y, color=None if color == "ninguno" else color, trend=tendencia), width="stretch")
        elif tipo == "Lineas" and numericas:
            if fechas:
                x = st.selectbox("Eje X (fecha)", fechas, key="sel_lineas_x_fecha")
            else:
                x = st.selectbox("Eje X", list(df.columns), key="sel_lineas_x")
            y = st.selectbox("Eje Y", numericas, key="sel_lineas_y")
            color = st.selectbox("Color (opcional)", ["ninguno"] + categoricas, key="sel_lineas_color")
            datos = df.sort_values(x)
            st.plotly_chart(pa.plots.interactive_line(datos, x, y, color=None if color == "ninguno" else color), width="stretch")
        elif tipo == "Barras" and categoricas and numericas:
            x = st.selectbox("Categoria", categoricas, key="sel_barras_x")
            y = st.selectbox("Valor", numericas, key="sel_barras_y")
            agregado = st.selectbox("Agregado", ["mean", "sum", "median", "count", "max", "min"], key="sel_barras_agg")
            datos = df.groupby(x, dropna=False)[y].agg(agregado).reset_index().sort_values(y, ascending=False).head(25)
            import plotly.express as px

            st.plotly_chart(px.bar(datos, x=x, y=y, text_auto=".2s"), width="stretch")
        elif tipo == "Frecuencia" and categoricas:
            col = st.selectbox("Columna", categoricas, key="sel_freq_col")
            vc = df[col].value_counts().head(25).reset_index()
            vc.columns = [col, "frecuencia"]
            import plotly.express as px

            st.plotly_chart(px.bar(vc, x=col, y="frecuencia", text_auto=True), width="stretch")
        elif tipo == "Circular" and categoricas:
            col = st.selectbox("Columna", categoricas, key="sel_pie_col")
            vc = df[col].value_counts().head(12).reset_index()
            vc.columns = [col, "frecuencia"]
            import plotly.express as px

            st.plotly_chart(px.pie(vc, names=col, values="frecuencia", hole=0.35), width="stretch")
        elif tipo == "Correlaciones" and len(numericas) >= 2:
            st.plotly_chart(pa.plots.interactive_corr(df), width="stretch")
        elif tipo == "Pares" and len(numericas) >= 2:
            st.pyplot(pa.pairplot(df, columns=numericas[:5]))
        else:
            st.info("Faltan columnas del tipo necesario para este grafico.")
    except Exception as exc:
        st.error(f"No se pudo generar el grafico: {exc}")

    with st.expander("Descargar un grafico PNG"):
        opcion = st.selectbox("Grafico", ["Histograma", "Caja", "Dispersion", "Correlaciones"], key="desc_graf")
        if opcion in ("Histograma", "Caja", "Dispersion") and not numericas:
            st.info("No hay columnas numericas para este grafico.")
            return
        if opcion in ("Dispersion", "Correlaciones") and len(numericas) < 2:
            st.info("Se necesitan al menos dos columnas numericas para este grafico.")
            return
        col = st.selectbox("Columna base", numericas, key="desc_col") if opcion != "Correlaciones" else None
        try:
            if opcion == "Histograma":
                fig = pa.histogram(df, col)
            elif opcion == "Caja":
                fig = pa.boxplot(df, col)
            elif opcion == "Dispersion":
                fig = pa.scatter(df, numericas[0], numericas[1], trend=True)
            else:
                fig = pa.heatmap_corr(df)
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            nombre = f"{opcion.lower()}" + (f"_{col}" if col else "") + ".png"
            st.download_button("Descargar PNG", buffer.getvalue(), file_name=nombre, mime="image/png", key="dl_png")
        except Exception as exc:
            st.caption(f"No disponible: {exc}")


def tab_series() -> None:
    st.header("6. Series temporales")
    df = _df_actual()
    if df is None:
        st.info("Primero carga datos.")
        return

    fechas = pa.datetime_columns(df)
    numericas = pa.numeric_columns(df)
    if not fechas:
        col = st.selectbox("Columna de fecha (texto)", list(df.columns), key="sel_fecha_texto")
        try:
            df = df.assign(**{col: pd.to_datetime(df[col], errors="coerce", format="mixed")}).dropna(subset=[col])
            fechas = [col]
        except Exception as exc:
            st.error(f"No se pudo interpretar la fecha: {exc}")
            return
    if not numericas:
        st.warning("No hay columnas numericas.")
        return

    c1, c2, c3 = st.columns(3)
    fecha = c1.selectbox("Fecha", fechas, key="sel_serie_fecha")
    valor = c2.selectbox("Valor", numericas, key="sel_serie_valor")
    freq = c3.selectbox("Frecuencia", ["D", "W", "MS", "QS", "YS"], index=2, format_func=lambda f: {"D": "Diaria", "W": "Semanal", "MS": "Mensual", "QS": "Trimestral", "YS": "Anual"}[f], key="sel_serie_freq")
    agregado = st.selectbox("Agregado", ["sum", "mean", "median", "max", "min", "count"], key="sel_serie_agg")

    try:
        serie_df = pa.to_datetime_index(df, fecha, valor, freq=freq, agg=agregado)
        serie = serie_df[valor]
    except Exception as exc:
        st.error(f"No se pudo construir la serie: {exc}")
        return

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=serie.index, y=serie.values, mode="lines+markers", name=valor))
    fig.update_layout(title=f"{valor} ({freq})", template="plotly_white")
    st.plotly_chart(fig, width="stretch")

    sub = st.tabs(["Descripcion", "Medias moviles", "Descomposicion", "Pronostico"])
    with sub[0]:
        info = pa.describe_series(serie)
        st.json(info)
        with st.expander("ACF y PACF"):
            st.pyplot(pa.acf_pacf_plot(serie))
    with sub[1]:
        ventanas = st.multiselect("Ventanas", [3, 7, 14, 30, 60], default=[7, 30], key="ms_ventanas")
        if ventanas:
            rolling = pa.rolling_stats(serie, windows=ventanas)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=serie.index, y=serie.values, name="original", opacity=0.5))
            for w in ventanas:
                fig2.add_trace(go.Scatter(x=serie.index, y=rolling[f"roll{w}_mean"], name=f"media {w}"))
            st.plotly_chart(fig2, width="stretch")
    with sub[2]:
        try:
            st.pyplot(pa.decompose(serie))
        except Exception as exc:
            st.info(f"No se pudo descomponer: {exc}")
    with sub[3]:
        horizonte = st.slider("Periodos a pronosticar", 1, 36, 12, key="slider_horizonte")
        metodo = st.selectbox("Metodo", ["auto", "naive", "seasonal_naive", "mean", "drift", "ses", "holt", "holt_winters", "sarima"], key="sel_serie_metodo")
        if st.button("Pronosticar", type="primary", key="btn_pronosticar"):
            try:
                pred = pa.forecast(serie, horizon=horizonte, method=metodo)
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=serie.index, y=serie.values, name="historico"))
                fig3.add_trace(go.Scatter(x=pred.index, y=pred["pronostico"], name="pronostico", line=dict(color="crimson")))
                if "limite_superior" in pred.columns:
                    fig3.add_trace(go.Scatter(x=pred.index, y=pred["limite_superior"], line=dict(width=0), showlegend=False))
                    fig3.add_trace(go.Scatter(x=pred.index, y=pred["limite_inferior"], fill="tonexty", line=dict(width=0), name="IC 95%"))
                st.plotly_chart(fig3, width="stretch")
                st.dataframe(pred.round(3), width="stretch")
                csv = pred.to_csv().encode("utf-8")
                st.download_button("Descargar pronostico CSV", csv, file_name="pronostico.csv", mime="text/csv", key="dl_pronostico")
            except Exception as exc:
                st.error(f"Error al pronosticar: {exc}")


def tab_modelar() -> None:
    st.header("7. Modelado")
    df = _df_actual()
    if df is None:
        st.info("Primero carga datos.")
        return

    objetivo = st.selectbox("Variable objetivo", list(df.columns), key="sel_objetivo")
    tarea_detectada = pa.detect_task(df[objetivo])
    tarea = st.radio("Tarea", ["auto", "classification", "regression"], index=0, horizontal=True, format_func=lambda t: f"{t} (detectada: {tarea_detectada})" if t == "auto" else t, key="radio_tarea")
    tarea_real = tarea_detectada if tarea == "auto" else tarea

    col1, col2, col3 = st.columns(3)
    modelos = col1.multiselect("Modelos a comparar", pa.list_models(tarea_real), default=pa.list_models(tarea_real)[:3], key="ms_modelos")
    cv = col2.slider("Particiones de validacion cruzada", 2, 10, 5, key="slider_cv")
    test_size = col3.slider("Tamano de prueba", 0.1, 0.4, 0.2, 0.05, key="slider_test_size")

    if st.button("Entrenar y comparar", type="primary", key="btn_entrenar") and modelos:
        with st.spinner("Entrenando modelos..."):
            try:
                comparacion = pa.compare_models(df, objetivo, models=modelos, task=tarea_real, cv=cv)
                st.subheader("Comparacion de modelos")
                st.dataframe(comparacion, width="stretch")
                criterio = "accuracy" if tarea_real == "classification" else "r2"
                if comparacion.empty or criterio not in comparacion.columns:
                    st.error(
                        "No se pudo entrenar ningun modelo con la configuracion actual. "
                        "Prueba con otra variable objetivo, menos particiones de validacion o mas filas."
                    )
                    return
                mejor = comparacion.sort_values(criterio, ascending=False).iloc[0]["modelo"]
                st.success(f"Mejor modelo segun {criterio}: **{mejor}**")

                resultado = pa.train_model(df, objetivo, model=mejor, task=tarea_real, test_size=test_size)
                st.subheader(f"Evaluacion de {mejor} en el conjunto de prueba")
                metricas = {k: v for k, v in resultado["metricas"].items() if not isinstance(v, dict)}
                cols = st.columns(min(len(metricas), 5))
                for i, (nombre, valor) in enumerate(metricas.items()):
                    cols[i % len(cols)].metric(nombre, valor)

                if tarea_real == "classification":
                    st.markdown("**Matriz de confusion**")
                    st.dataframe(pa.confusion_matrix_df(resultado["y_test"], resultado["y_pred"]), width="stretch")
                else:
                    st.markdown("**Residuos**")
                    st.dataframe(pa.residuals(resultado["y_test"], resultado["y_pred"]).describe().round(3), width="stretch")

                with st.expander("Importancia de variables (permutacion)"):
                    imp = pa.permutation_importance(resultado["pipeline"], resultado["X_test"], resultado["y_test"])
                    st.dataframe(imp, width="stretch")
                    import plotly.express as px

                    st.plotly_chart(px.bar(imp.head(15), x="importancia_media", y="caracteristica", orientation="h"), width="stretch")
            except Exception as exc:
                st.error(f"Error en el modelado: {exc}")

    st.divider()
    with st.expander("Modelo base automatico con skrub (opcional)"):
        st.caption(
            "skrub imputa nulos, codifica texto y fechas y escala sin preprocesar: "
            "util como **linea base** frente a los modelos de arriba."
        )
        if not pa.skrub_disponible():
            st.info("Instala las dependencias opcionales para habilitarlo: `pip install -r requirements-optional.txt`.")
        elif st.button("Entrenar modelo base con skrub", key="btn_skrub"):
            with st.spinner("Entrenando modelo base..."):
                try:
                    res = pa.modelo_rapido(df, objetivo, tarea=tarea_real, test_size=test_size)
                    st.success(f"Modelo: {res['modelo']} · tarea: {res['tarea']} · prueba: {res['n_prueba']} filas")
                    metricas = {k: v for k, v in res["metricas"].items() if not isinstance(v, dict)}
                    cols = st.columns(min(len(metricas), 5))
                    for i, (nombre, valor) in enumerate(metricas.items()):
                        cols[i % len(cols)].metric(nombre, valor)
                except Exception as exc:
                    st.error(f"No se pudo entrenar el modelo base: {exc}")


def tab_informe() -> None:
    st.header("8. Exportar")
    df = _df_actual()
    if df is None:
        st.info("Primero carga datos.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Datos procesados")
        formato = st.selectbox("Formato", ["csv", "xlsx", "parquet", "json"], key="sel_formato")
        if formato == "csv":
            st.download_button("Descargar CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name="datos_procesados.csv", mime="text/csv", width="stretch", key="dl_csv")
        elif formato == "xlsx":
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="datos")
            st.download_button("Descargar Excel", buffer.getvalue(), file_name="datos_procesados.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch", key="dl_xlsx")
        elif formato == "parquet":
            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False)
            st.download_button("Descargar Parquet", buffer.getvalue(), file_name="datos_procesados.parquet", mime="application/octet-stream", width="stretch", key="dl_parquet")
        else:
            st.download_button("Descargar JSON", df.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8"), file_name="datos_procesados.json", mime="application/json", width="stretch", key="dl_json")

        if "informe_limpieza" in st.session_state:
            st.text_area("Registro de limpieza", st.session_state["informe_limpieza"], height=160, key="ta_registro_limpieza")

    with col2:
        st.subheader("Informe HTML")
        titulo = st.text_input("Titulo del informe", "Informe de datos", key="txt_titulo_informe")
        incluir = st.checkbox("Incluir graficos", value=True, key="chk_incluir_graficos")
        if st.button("Generar informe", type="primary", width="stretch", key="btn_generar_informe"):
            with st.spinner("Generando informe..."):
                try:
                    destino = RAIZ / "outputs" / "informe.html"
                    ruta = pa.generate_report(df, destino, title=titulo, include_plots=incluir)
                    st.session_state["html_informe"] = Path(ruta).read_text(encoding="utf-8")
                    st.success("Informe generado.")
                except Exception as exc:
                    st.error(f"Error al generar el informe: {exc}")
        if "html_informe" in st.session_state:
            st.download_button("Descargar informe HTML", st.session_state["html_informe"].encode("utf-8"), file_name="informe.html", mime="text/html", width="stretch", key="dl_informe_html_tab8")

        st.divider()
        st.subheader("Informe exploratorio (skrub)")
        st.caption(
            "Perfilado interactivo con skrub."
            if pa.skrub_disponible()
            else "skrub no esta instalado: se usara el informe propio del paquete."
        )
        if st.button("Generar informe exploratorio", width="stretch", key="btn_eda_tab8"):
            with st.spinner("Generando informe exploratorio..."):
                try:
                    st.session_state["html_eda"] = pa.informe_eda_html(df, titulo=titulo)
                    st.success("Informe exploratorio generado.")
                except Exception as exc:
                    st.error(f"Error al generar el informe exploratorio: {exc}")
        if st.session_state.get("html_eda"):
            st.download_button("Descargar informe exploratorio", st.session_state["html_eda"].encode("utf-8"), file_name="perfil_exploratorio.html", mime="text/html", width="stretch", key="dl_eda_tab8")


# --------------------------------------------------------------------------- #
def main() -> None:
    barra_lateral()
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-badge">🌿 pyanalytics · v{pa.__version__}</div>
          <h1>Analitica y exploracion de datos con Python</h1>
          <p>Carga, limpia, explora, grafica, modela y exporta datos de ingenieria ambiental
          en Colombia, sin escribir codigo.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    pestanas = st.tabs(
        ["📁 Cargar", "🔍 Explorar", "🧹 Limpiar", "⚖️ Normalizar", "📊 Graficar", "📈 Series", "🤖 Modelar", "💾 Exportar"]
    )
    with pestanas[0]:
        tab_carga()
    with pestanas[1]:
        tab_explorar()
    with pestanas[2]:
        tab_limpiar()
    with pestanas[3]:
        tab_normalizar()
    with pestanas[4]:
        tab_graficar()
    with pestanas[5]:
        tab_series()
    with pestanas[6]:
        tab_modelar()
    with pestanas[7]:
        tab_informe()

    st.markdown('<div class="pie">pyanalytics · analitica exploratoria con Python y Streamlit</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
