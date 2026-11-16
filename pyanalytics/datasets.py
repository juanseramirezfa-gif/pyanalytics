"""Datasets de ejemplo (sinteticos) de ingenieria ambiental en Colombia.

Sirven para probar la herramienta sin datos reales. Incluyen a proposito
nulos, duplicados, texto "sucio", valores atipicos y variables objetivo para
modelado, de modo que los ejemplos y la app web muestren casos realistas.

Datasets disponibles
--------------------
- ``calidad_agua``: monitoreo de calidad del agua (rios, quebradas, humedales)
  con el Indice de Calidad del Agua (ICA) y el cumplimiento de norma.
- ``aire_mensual``: serie temporal mensual de material particulado (PM2.5 y
  PM10) en estaciones de monitoreo del aire.

Ejemplos
--------
>>> from pyanalytics.datasets import calidad_agua, aire_mensual
>>> df = calidad_agua()
>>> df.shape
(815, 17)
>>> serie = aire_mensual()
>>> sorted(serie.columns)
['estacion', 'fecha', 'pm10_ug_m3', 'pm25_ug_m3']
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

__all__ = [
    "calidad_agua",
    "aire_mensual",
    "DATASETS",
    "lista_datasets",
    "cargar_ejemplo",
]


def _formato_es(valor: Any) -> str | None:
    """Formatea un numero al estilo espanol (1.234,56)."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return None
    return f"{float(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _subindice_mayor_mejor(x: np.ndarray, malo: float, bueno: float) -> np.ndarray:
    """Subindice 0-100 donde valores altos son mejores (p. ej. oxigeno disuelto)."""
    return np.clip((x - malo) / (bueno - malo) * 100.0, 0.0, 100.0)


def _subindice_menor_mejor(x: np.ndarray, bueno: float, malo: float) -> np.ndarray:
    """Subindice 0-100 donde valores bajos son mejores (p. ej. DBO5, SST)."""
    return np.clip((malo - x) / (malo - bueno) * 100.0, 0.0, 100.0)


def calidad_agua(n: int = 800, semilla: int = 7, formato_sucio: bool = True) -> pd.DataFrame:
    """Genera un dataset de monitoreo de calidad del agua en Colombia.

    Parametros
    ----------
    n:
        Numero de muestras base (antes de anadir duplicados).
    semilla:
        Semilla del generador aleatorio, para resultados reproducibles.
    formato_sucio:
        Si es ``True`` anade nulos, duplicados, texto sucio y atipicos.

    Columnas
    --------
    ``id_muestra``, ``Fecha Muestreo``, ``Departamento``, ``Cuenca``,
    ``Tipo Fuente``, ``Caudal (m3/s)``, ``pH``, ``Oxigeno Disuelto (mg/L)``,
    ``DBO5 (mg/L)``, ``DQO (mg/L)``, ``Solidos Suspendidos (mg/L)``,
    ``Turbidez (NTU)``, ``Temperatura Agua (C)``, ``Conductividad (us/cm)``,
    ``Coliformes Fecales (UFC/100ml)``, ``Indice ICA`` y ``Cumplimiento Norma``.

    El ``Indice ICA`` (0-100, mayor es mejor) se calcula a partir de los
    subindices de oxigeno disuelto, DBO5, solidos suspendidos, pH y
    coliformes, siguiendo la logica del ICA-IDEAM. ``Cumplimiento Norma``
    vale ``"cumple"`` o ``"no cumple"`` segun el ICA.
    """
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range("2022-01-01", periods=n, freq="D")

    departamentos = np.array(
        ["Antioquia", "Cundinamarca", "Valle del Cauca", "Atlantico", "Bolivar", "Santander", "Boyaca"],
        dtype=object,
    )
    cuencas = np.array(
        ["Rio Magdalena", "Rio Cauca", "Rio Bogota", "Rio Medellin", "Cienaga Grande", "Rio Sogamoso", "Lago de Tota"],
        dtype=object,
    )
    tipos = np.array(["Rio", "Quebrada", "Humedal", "Embalse", "Canal", "Nacimiento"], dtype=object)

    od = np.clip(rng.normal(6.6, 1.6, n), 1.0, 11.5)
    dbo5 = np.clip(rng.gamma(2.0, 6.0, n), 0.8, 70.0)
    dqo = dbo5 * rng.normal(2.3, 0.25, n)
    sst = np.clip(rng.gamma(2.0, 26.0, n), 2.0, 320.0)
    turbidez = np.clip(sst * rng.normal(1.5, 0.2, n) + rng.normal(0, 5, n), 0.5, None)
    ph = np.clip(rng.normal(7.2, 0.7, n), 4.0, 9.8)
    temperatura = np.clip(rng.normal(24.0, 3.0, n), 12.0, 33.0)
    caudal = np.clip(rng.gamma(2.0, 8.0, n), 0.4, 260.0)
    conductividad = np.clip(rng.normal(260.0, 85.0, n), 30.0, 900.0)
    coliformes = 10 ** np.clip(rng.normal(3.0, 1.3, n), 1.0, 6.2)  # UFC/100 mL

    # Indice de Calidad del Agua (aprox. ICA-IDEAM): 0-100, mayor es mejor.
    score_od = _subindice_mayor_mejor(od, 2.0, 9.0)
    score_dbo = _subindice_menor_mejor(dbo5, 2.0, 20.0)
    score_sst = _subindice_menor_mejor(sst, 10.0, 100.0)
    score_ph = np.clip(100.0 - np.abs(ph - 7.5) * 35.0, 0.0, 100.0)
    score_coli = _subindice_menor_mejor(np.log10(coliformes), 2.0, 5.0)

    ica = (
        0.25 * score_od
        + 0.25 * score_dbo
        + 0.15 * score_sst
        + 0.15 * score_ph
        + 0.20 * score_coli
    )
    ica = np.clip(ica + rng.normal(0.0, 2.5, n), 0.0, 100.0)

    cumple = np.where(ica >= 65.0, "cumple", "no cumple").astype(object)
    voltear = rng.random(n) < 0.04  # ruido de etiqueta para que no sea trivial
    cumple = np.where(voltear, np.where(cumple == "cumple", "no cumple", "cumple"), cumple).astype(object)
    cumple = np.where(rng.random(n) < 0.05, None, cumple).astype(object)

    df = pd.DataFrame(
        {
            "id_muestra": np.arange(1000, 1000 + n),
            "Fecha Muestreo": fechas.astype(str),
            "Departamento ": departamentos[rng.integers(0, len(departamentos), n)],
            "Cuenca": cuencas[rng.integers(0, len(cuencas), n)],
            "Tipo Fuente": tipos[rng.integers(0, len(tipos), n)],
            "Caudal (m3/s)": np.round(caudal, 3),
            "pH": np.round(ph, 2),
            "Oxigeno Disuelto (mg/L)": np.round(od, 2),
            "DBO5 (mg/L)": np.round(dbo5, 2),
            "DQO (mg/L)": np.round(dqo, 2),
            "Solidos Suspendidos (mg/L)": np.round(sst, 2),
            "Turbidez (NTU)": np.round(turbidez, 2),
            "Temperatura Agua (C)": np.round(temperatura, 2),
            "Conductividad (us/cm)": np.round(conductividad, 1),
            "Coliformes Fecales (UFC/100ml)": np.round(coliformes, 1),
            "Indice ICA": np.round(ica, 2),
            "Cumplimiento Norma": cumple,
        }
    )

    if formato_sucio:
        # numeros guardados como texto con formato espanol (1.234,56)
        df["Solidos Suspendidos (mg/L)"] = df["Solidos Suspendidos (mg/L)"].map(_formato_es)

        def _filas(k: int) -> np.ndarray:
            """Posiciones aleatorias sin repeticion (se adapta a n pequeno)."""
            return rng.choice(n, min(k, n), replace=False)

        # valores nulos
        df.loc[_filas(30), "Oxigeno Disuelto (mg/L)"] = np.nan
        df.loc[_filas(18), "Solidos Suspendidos (mg/L)"] = np.nan
        df.loc[_filas(14), "Tipo Fuente"] = None
        # valores atipicos (posibles errores de medicion)
        df.loc[_filas(8), "DBO5 (mg/L)"] = 999.0
        df.loc[_filas(5), "Turbidez (NTU)"] = 5000.0
        # categorias sucias (mayusculas/minusculas y espacios)
        sucios = _filas(25)
        df.loc[sucios, "Departamento "] = "  " + df.loc[sucios, "Departamento "].astype(str).str.lower() + " "
        # filas duplicadas
        df = pd.concat([df, df.iloc[: min(15, n)]], ignore_index=True)

    return df


def aire_mensual(
    meses: int = 60,
    semilla: int = 21,
    estaciones: tuple[str, ...] = ("Estacion Centro", "Estacion Norte"),
) -> pd.DataFrame:
    """Genera una serie temporal mensual de calidad del aire (PM2.5 y PM10).

    Simula el ciclo bimodal de epocas secas/humedas de Colombia (picos hacia
    enero-febrero y julio-agosto) mas una tendencia suave, para varias
    estaciones de monitoreo.

    Columnas: ``fecha``, ``pm25_ug_m3``, ``pm10_ug_m3`` y ``estacion``.
    """
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range("2020-01-01", periods=meses, freq="MS")
    t = np.arange(meses)
    estacional = 7.0 * np.cos(2 * np.pi * t / 12) + 4.0 * np.cos(4 * np.pi * t / 12)

    frames: list[pd.DataFrame] = []
    for i, estacion in enumerate(estaciones):
        base = 22.0 + 3.0 * i
        pm25 = base + 0.06 * t + estacional + rng.normal(0.0, 2.0, meses)
        pm10 = pm25 * 1.6 + rng.normal(0.0, 4.0, meses)
        frames.append(
            pd.DataFrame(
                {
                    "fecha": fechas,
                    "pm25_ug_m3": np.round(np.clip(pm25, 6.0, None), 2),
                    "pm10_ug_m3": np.round(np.clip(pm10, 9.0, None), 2),
                    "estacion": estacion,
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)
    total = len(df)
    df.loc[rng.choice(total, 3, replace=False), "pm10_ug_m3"] = np.nan
    return df


#: Registro de datasets de ejemplo disponibles.
DATASETS: dict[str, Callable[..., pd.DataFrame]] = {
    "calidad_agua": calidad_agua,
    "aire_mensual": aire_mensual,
}


def lista_datasets() -> list[str]:
    """Nombres de los datasets de ejemplo disponibles."""
    return sorted(DATASETS)


def cargar_ejemplo(nombre: str = "calidad_agua", **kwargs: Any) -> pd.DataFrame:
    """Devuelve un dataset de ejemplo por su nombre.

    >>> cargar_ejemplo("calidad_agua", n=100).shape
    (115, 17)
    """
    if nombre not in DATASETS:
        raise ValueError(f"Dataset desconocido: {nombre!r}. Disponibles: {lista_datasets()}")
    return DATASETS[nombre](**kwargs)
