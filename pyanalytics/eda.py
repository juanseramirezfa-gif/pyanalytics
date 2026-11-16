"""Perfilado exploratorio interactivo y pipelines automaticos (integracion skrub).

Este modulo conecta pyanalytics con `skrub <https://skrub-data.org>`_, la
libreria de *machine learning with dataframes* mantenida por INRIA (licencia
BSD-3). Aporta tres cosas que el paquete no tenia:

- un **informe de perfilado exploratorio interactivo** y autocontenido
  (:func:`informe_eda_html`, :func:`guardar_informe_eda`);
- una **tabla de asociaciones** entre pares de columnas que, a diferencia de
  la matriz de correlacion clasica, tambien cubre variables categoricas
  (V de Cramer) (:func:`asociaciones_columnas`);
- un **modelo base en una linea** que maneja texto, nulos y fechas sin
  preprocesar (:func:`modelo_rapido`).

skrub es una dependencia **opcional**: si no esta instalada, el informe cae
automaticamente al generador propio del paquete y las funciones que la
necesitan lanzan un mensaje claro. Para habilitarla:

.. code-block:: powershell

    .venv\\Scripts\\python.exe -m pip install -r requirements-optional.txt

Ejemplos
--------
>>> import pyanalytics as pa
>>> pa.skrub_disponible()
True
>>> html = pa.informe_eda_html(df, titulo="Calidad del agua")
>>> pa.guardar_informe_eda(df, "outputs/eda.html")
>>> pa.modelo_rapido(df, objetivo="indice_ica")["metricas"]["r2"]
"""

from __future__ import annotations

import tempfile
from html import escape
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .modeling import detect_task, evaluate_classification, evaluate_regression, split_data
from .profile import correlation_matrix

__all__ = [
    "skrub_disponible",
    "version_skrub",
    "informe_eda_html",
    "guardar_informe_eda",
    "asociaciones_columnas",
    "modelo_rapido",
]

_MENSAJE_SKRUB = (
    "Esta funcion necesita la libreria opcional 'skrub'. "
    "Instalala con:  pip install -r requirements-optional.txt"
)

Task = Literal["auto", "classification", "regression"]


def skrub_disponible() -> bool:
    """Indica si la libreria opcional ``skrub`` esta instalada."""
    try:
        import skrub  # noqa: F401
    except Exception:
        return False
    return True


def version_skrub() -> str | None:
    """Version de skrub instalada, o ``None`` si no esta disponible."""
    try:
        import skrub
    except Exception:
        return None
    return getattr(skrub, "__version__", None)


def informe_eda_html(df: pd.DataFrame, *, titulo: str = "Informe exploratorio") -> str:
    """Devuelve un informe de perfilado exploratorio como HTML autocontenido.

    Con skrub instalado usa su ``TableReport``: un documento interactivo con
    distribuciones, valores representativos, nulos y asociaciones. Sin skrub
    cae al generador propio del paquete (:func:`pyanalytics.generate_report`).
    """
    if df is None or df.empty:
        raise ValueError("El DataFrame esta vacio.")
    if skrub_disponible():
        import skrub

        contenido = skrub.TableReport(df).html()
        return _titular(contenido, titulo)
    return _informe_propio(df, titulo)


def guardar_informe_eda(
    df: pd.DataFrame,
    ruta: str | Path = "outputs/perfil_exploratorio.html",
    *,
    titulo: str = "Informe exploratorio",
) -> Path:
    """Escribe el informe exploratorio en ``ruta`` y devuelve la ruta."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(informe_eda_html(df, titulo=titulo), encoding="utf-8")
    return ruta


def asociaciones_columnas(df: pd.DataFrame, *, compute_pearson: bool = True) -> pd.DataFrame:
    """Tabla de asociaciones entre pares de columnas, ordenada por fuerza.

    Con skrub devuelve la tabla larga de :func:`skrub.column_associations`
    (columnas ``left_column_name``, ``right_column_name``, ``cramer_v`` y
    ``pearson_corr``) ordenada de mayor a menor asociacion, de modo que sirve
    tambien para variables categoricas. Sin skrub devuelve la matriz de
    correlacion numerica del paquete (:func:`pyanalytics.correlation_matrix`).
    """
    if not skrub_disponible():
        return correlation_matrix(df)

    import skrub

    tabla = skrub.column_associations(df, compute_pearson=compute_pearson).copy()
    cramer = tabla["cramer_v"].abs().fillna(0.0) if "cramer_v" in tabla else 0.0
    pearson = tabla["pearson_corr"].abs().fillna(0.0) if "pearson_corr" in tabla else 0.0
    tabla["_fuerza"] = pd.concat([cramer, pearson], axis=1).max(axis=1)
    tabla = tabla.sort_values("_fuerza", ascending=False).drop(columns="_fuerza")
    return tabla.reset_index(drop=True)


def modelo_rapido(
    df: pd.DataFrame,
    objetivo: str,
    *,
    tarea: Task = "auto",
    test_size: float = 0.2,
    semilla: int = 0,
) -> dict[str, Any]:
    """Entrena un modelo base con skrub y devuelve metricas y predicciones.

    skrub imputa nulos, codifica texto y fechas y escala de forma automatica,
    por lo que sirve como **linea base** para cualquier DataFrame tabular. La
    tarea (clasificacion o regresion) se detecta sola salvo que se indique con
    ``tarea=``. Devuelve un diccionario con ``tarea``, ``modelo``, ``metricas``,
    ``pipeline``, ``y_test``, ``y_pred`` y los tamanos de entrenamiento/prueba.
    """
    if not skrub_disponible():
        raise ImportError(_MENSAJE_SKRUB)
    if objetivo not in df.columns:
        raise KeyError(f"El objetivo {objetivo!r} no existe en el DataFrame.")

    import skrub

    datos = df.dropna(subset=[objetivo]).copy()
    if datos.empty:
        raise ValueError("No quedan filas tras descartar los nulos del objetivo.")

    tipo: Literal["classification", "regression"] = detect_task(datos[objetivo]) if tarea == "auto" else tarea
    particion = split_data(datos, objetivo, test_size=test_size, random_state=semilla, task=tipo)

    estimador = "classifier" if tipo == "classification" else "regressor"
    pipeline = skrub.tabular_pipeline(estimador)
    pipeline.fit(particion["X_train"], particion["y_train"])
    y_pred = pipeline.predict(particion["X_test"])

    if tipo == "classification":
        metricas = evaluate_classification(particion["y_test"], y_pred)
    else:
        metricas = evaluate_regression(particion["y_test"], y_pred)

    return {
        "tarea": tipo,
        "modelo": f"skrub.tabular_pipeline({estimador!r})",
        "metricas": metricas,
        "pipeline": pipeline,
        "y_test": particion["y_test"],
        "y_pred": y_pred,
        "n_entrenamiento": int(len(particion["X_train"])),
        "n_prueba": int(len(particion["X_test"])),
    }


# --------------------------------------------------------------------------- #
# ayudantes internos
# --------------------------------------------------------------------------- #
def _titular(contenido: str, titulo: str) -> str:
    """Sustituye el titulo por defecto del informe de skrub."""
    return contenido.replace("<title>skrub report</title>", f"<title>{escape(titulo)}</title>", 1)


def _informe_propio(df: pd.DataFrame, titulo: str) -> str:
    """Genera el informe con el motor propio del paquete (sin skrub)."""
    from .report import generate_report

    with tempfile.TemporaryDirectory() as carpeta:
        ruta = generate_report(df, Path(carpeta) / "informe.html", title=titulo, include_plots=True)
        return Path(ruta).read_text(encoding="utf-8")
