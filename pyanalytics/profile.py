"""Exploracion y perfilado de un DataFrame.

Todas las funciones son puras: reciben un DataFrame y devuelven tablas o
diccionarios, sin modificar los datos.

Ejemplos
--------
>>> from pyanalytics.profile import full_profile, column_summary
>>> column_summary(df)
>>> perfil = full_profile(df)
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

__all__ = [
    "numeric_columns",
    "categorical_columns",
    "text_columns",
    "datetime_columns",
    "column_types",
    "overview",
    "column_summary",
    "missing_report",
    "duplicates_report",
    "numeric_stats",
    "categorical_stats",
    "correlation_matrix",
    "high_correlations",
    "detect_outliers",
    "outliers_report",
    "value_counts_table",
    "data_quality_score",
    "full_profile",
]

OutlierMethod = Literal["iqr", "zscore", "mad"]


def numeric_columns(df: pd.DataFrame, include_bool: bool = False) -> list[str]:
    """Nombres de las columnas numericas."""
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not include_bool:
        cols = [c for c in cols if df[c].dtype != bool]
    return cols


def text_columns(df: pd.DataFrame) -> list[str]:
    """Nombres de las columnas de texto.

    Funciona igual con pandas 2 (dtype ``object``) y pandas 3 (dtype ``str``).
    """
    return [
        c
        for c in df.columns
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
    ]


def categorical_columns(df: pd.DataFrame, max_unique: int | None = None) -> list[str]:
    """Nombres de las columnas categoricas (texto, categoria o booleano)."""
    cols = [
        c
        for c in df.columns
        if pd.api.types.is_bool_dtype(df[c])
        or isinstance(df[c].dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(df[c])
        or pd.api.types.is_string_dtype(df[c])
    ]
    if max_unique is not None:
        cols = [c for c in cols if df[c].nunique(dropna=True) <= max_unique]
    return cols


def datetime_columns(df: pd.DataFrame) -> list[str]:
    """Nombres de las columnas de fecha/hora ya tipadas."""
    return df.select_dtypes(include=["datetime", "datetimetz", "timedelta"]).columns.tolist()


def column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """Agrupa las columnas por tipo logico."""
    num = numeric_columns(df)
    dt = datetime_columns(df)
    cat = [c for c in categorical_columns(df) if c not in num and c not in dt]
    otras = [c for c in df.columns if c not in num and c not in dt and c not in cat]
    return {"numericas": num, "fechas": dt, "categoricas": cat, "otras": otras}


def overview(df: pd.DataFrame) -> dict[str, Any]:
    """Resumen general: dimensiones, memoria, tipos y duplicados."""
    memory_bytes = int(df.memory_usage(deep=True).sum())
    tipos = column_types(df)
    return {
        "filas": int(df.shape[0]),
        "columnas": int(df.shape[1]),
        "memoria_bytes": memory_bytes,
        "memoria_legible": _human_bytes(memory_bytes),
        "n_columnas_numericas": len(tipos["numericas"]),
        "n_columnas_fechas": len(tipos["fechas"]),
        "n_columnas_categoricas": len(tipos["categoricas"]),
        "n_columnas_otras": len(tipos["otras"]),
        "celdas_totales": int(df.shape[0] * df.shape[1]),
        "celdas_nulas": int(df.isna().sum().sum()),
        "porcentaje_nulos": round(float(df.isna().mean().mean() * 100), 3),
        "filas_duplicadas": int(df.duplicated().sum()),
        "columnas_con_nulos": int((df.isna().any()).sum()),
        "columnas_constantes": int(sum(df[c].nunique(dropna=False) <= 1 for c in df.columns)),
    }


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por columna con su diagnostico basico."""
    rows: list[dict[str, Any]] = []
    n = max(len(df), 1)
    for col in df.columns:
        serie = df[col]
        nulos = int(serie.isna().sum())
        unicos = int(serie.nunique(dropna=True))
        row: dict[str, Any] = {
            "columna": col,
            "tipo": str(serie.dtype),
            "tipo_logico": _logical_kind(serie),
            "no_nulos": int(serie.notna().sum()),
            "nulos": nulos,
            "pct_nulos": round(nulos / n * 100, 3),
            "unicos": unicos,
            "pct_unicos": round(unicos / n * 100, 3),
            "es_constante": unicos <= 1,
            "ejemplo": _first_value(serie),
        }
        if pd.api.types.is_numeric_dtype(serie) and serie.notna().any():
            row.update(
                {
                    "min": _safe(serie.min()),
                    "max": _safe(serie.max()),
                    "media": _safe(serie.mean()),
                    "mediana": _safe(serie.median()),
                    "desv": _safe(serie.std()),
                    "ceros": int((serie == 0).sum()),
                    "negativos": int((serie < 0).sum()),
                }
            )
        if pd.api.types.is_datetime64_any_dtype(serie) and serie.notna().any():
            row.update(
                {
                    "min": _safe(serie.min()),
                    "max": _safe(serie.max()),
                    "rango_dias": _safe((serie.max() - serie.min()).days),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de valores nulos por columna, ordenada de mayor a menor."""
    total = len(df)
    data = {
        "columna": df.columns,
        "nulos": [int(df[c].isna().sum()) for c in df.columns],
        "pct_nulos": [round(float(df[c].isna().mean() * 100), 3) for c in df.columns],
        "tipo": [str(df[c].dtype) for c in df.columns],
        "no_nulos": [int(df[c].notna().sum()) for c in df.columns],
    }
    rep = pd.DataFrame(data)
    rep["total_filas"] = total
    return rep.sort_values("nulos", ascending=False, ignore_index=True)


def duplicates_report(df: pd.DataFrame, subset: list[str] | None = None) -> dict[str, Any]:
    """Cuantas filas duplicadas hay y cuales son."""
    mask = df.duplicated(subset=subset, keep=False)
    duplicadas = df[mask]
    return {
        "n_filas_duplicadas_exactas": int(df.duplicated(subset=subset).sum()),
        "n_filas_involucradas": int(mask.sum()),
        "porcentaje": round(float(mask.mean() * 100), 3),
        "columnas_comparadas": subset or list(df.columns),
        "muestra": duplicadas.head(20),
    }


def numeric_stats(df: pd.DataFrame, percentiles: tuple[float, ...] = (0.01, 0.25, 0.5, 0.75, 0.99)) -> pd.DataFrame:
    """Estadisticos descriptivos de las columnas numericas (transpuesto)."""
    cols = numeric_columns(df)
    if not cols:
        return pd.DataFrame()
    desc = df[cols].describe(percentiles=list(percentiles)).T
    desc["asimetria"] = df[cols].skew(numeric_only=True)
    desc["curtosis"] = df[cols].kurtosis(numeric_only=True)
    desc["nulos"] = df[cols].isna().sum()
    return desc.round(4)


def categorical_stats(df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Resumen de columnas categoricas: cardinalidad y moda."""
    cols = categorical_columns(df)
    rows: list[dict[str, Any]] = []
    for col in cols:
        serie = df[col]
        vc = serie.value_counts(dropna=True)
        moda = vc.index[0] if len(vc) else None
        rows.append(
            {
                "columna": col,
                "unicos": int(serie.nunique(dropna=True)),
                "moda": moda,
                "frecuencia_moda": int(vc.iloc[0]) if len(vc) else 0,
                "pct_moda": round(float(vc.iloc[0] / max(len(serie), 1) * 100), 2) if len(vc) else 0.0,
                "nulos": int(serie.isna().sum()),
                "top_valores": ", ".join(map(str, vc.head(top).index.tolist())),
            }
        )
    return pd.DataFrame(rows)


def correlation_matrix(
    df: pd.DataFrame,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    min_periods: int = 1,
) -> pd.DataFrame:
    """Matriz de correlacion entre columnas numericas."""
    cols = numeric_columns(df)
    if len(cols) < 2:
        return pd.DataFrame()
    return df[cols].corr(method=method, min_periods=min_periods).round(4)


def high_correlations(
    df: pd.DataFrame,
    threshold: float = 0.8,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
) -> pd.DataFrame:
    """Pares de columnas con correlacion absoluta por encima del umbral."""
    corr = correlation_matrix(df, method=method)
    if corr.empty:
        return pd.DataFrame(columns=["columna_a", "columna_b", "correlacion"])
    pairs: list[dict[str, Any]] = []
    cols = corr.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            value = corr.loc[a, b]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append({"columna_a": a, "columna_b": b, "correlacion": round(float(value), 4)})
    out = pd.DataFrame(pairs)
    if not out.empty:
        out = out.reindex(out["correlacion"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return out


def detect_outliers(
    df: pd.DataFrame,
    column: str,
    method: OutlierMethod = "iqr",
    threshold: float = 1.5,
) -> pd.Series:
    """Marca con ``True`` los valores atipicos de una columna numerica.

    ``method``: ``"iqr"`` (rango intercuartilico), ``"zscore"`` o ``"mad"``.
    """
    serie = pd.to_numeric(df[column], errors="coerce")
    if method == "iqr":
        q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            return pd.Series(False, index=df.index)
        low, high = q1 - threshold * iqr, q3 + threshold * iqr
        return (serie < low) | (serie > high)
    if method == "zscore":
        std = serie.std()
        if not std or pd.isna(std):
            return pd.Series(False, index=df.index)
        return ((serie - serie.mean()) / std).abs() > threshold
    if method == "mad":
        med = serie.median()
        mad = (serie - med).abs().median()
        if not mad or pd.isna(mad):
            return pd.Series(False, index=df.index)
        return (0.6745 * (serie - med) / mad).abs() > threshold
    raise ValueError("method debe ser 'iqr', 'zscore' o 'mad'.")


def outliers_report(
    df: pd.DataFrame,
    method: OutlierMethod = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Resumen de atipicos para todas las columnas numericas."""
    columnas = ["columna", "metodo", "n_atipicos", "pct_atipicos", "limite_inferior", "limite_superior"]
    rows: list[dict[str, Any]] = []
    for col in numeric_columns(df):
        mask = detect_outliers(df, col, method=method, threshold=threshold)
        n = int(mask.sum())
        rows.append(
            {
                "columna": col,
                "metodo": method,
                "n_atipicos": n,
                "pct_atipicos": round(n / max(len(df), 1) * 100, 3),
                "limite_inferior": _safe(_bounds(df[col], method, threshold)[0]),
                "limite_superior": _safe(_bounds(df[col], method, threshold)[1]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columnas)
    return pd.DataFrame(rows).sort_values("n_atipicos", ascending=False, ignore_index=True)


def value_counts_table(
    df: pd.DataFrame,
    column: str,
    top: int | None = 20,
    normalize: bool = True,
) -> pd.DataFrame:
    """Frecuencias de una columna (categorica o discretizada)."""
    vc = df[column].value_counts(dropna=False).head(top)
    out = vc.rename("frecuencia").to_frame()
    if normalize:
        out["porcentaje"] = (out["frecuencia"] / max(len(df), 1) * 100).round(3)
    out["acumulado_%"] = out["porcentaje"].cumsum().round(3) if normalize else np.nan
    out.index.name = column
    return out.reset_index()


def data_quality_score(df: pd.DataFrame) -> dict[str, Any]:
    """Puntaje 0-100 de calidad de datos con los problemas detectados."""
    n_cells = max(df.shape[0] * df.shape[1], 1)
    problemas: list[str] = []

    pct_nulos = float(df.isna().sum().sum() / n_cells * 100)
    pct_dup = float(df.duplicated().mean() * 100) if len(df) else 0.0
    constantes = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    altas_card = [c for c in categorical_columns(df) if df[c].nunique(dropna=True) > 0.5 * max(len(df), 1)]

    puntaje = 100.0
    puntaje -= min(pct_nulos * 1.5, 40)
    puntaje -= min(pct_dup * 1.5, 25)
    puntaje -= min(len(constantes) * 3, 15)

    if pct_nulos > 5:
        problemas.append(f"{pct_nulos:.1f}% de celdas nulas")
    if pct_dup > 0:
        problemas.append(f"{pct_dup:.1f}% de filas duplicadas")
    if constantes:
        problemas.append(f"{len(constantes)} columna(s) constante(s): {', '.join(constantes[:5])}")
    if altas_card:
        problemas.append(f"alta cardinalidad en: {', '.join(altas_card[:5])}")
    numericas_sin_nulos = [c for c in numeric_columns(df) if df[c].notna().any()]
    muy_asimetricas = [c for c in numericas_sin_nulos if abs(float(df[c].skew())) > 3]
    if muy_asimetricas:
        problemas.append(f"distribucion muy asimetrica en: {', '.join(muy_asimetricas[:5])}")

    return {
        "puntaje": round(max(puntaje, 0.0), 1),
        "pct_celdas_nulas": round(pct_nulos, 3),
        "pct_filas_duplicadas": round(pct_dup, 3),
        "columnas_constantes": constantes,
        "problemas": problemas,
    }


def full_profile(df: pd.DataFrame, corr_threshold: float = 0.8) -> dict[str, Any]:
    """Perfil completo: todo lo anterior en un solo diccionario."""
    return {
        "overview": overview(df),
        "tipos": column_types(df),
        "columnas": column_summary(df),
        "nulos": missing_report(df),
        "duplicados": duplicates_report(df),
        "estadisticos_numericos": numeric_stats(df),
        "estadisticos_categoricos": categorical_stats(df),
        "correlaciones": correlation_matrix(df),
        "correlaciones_altas": high_correlations(df, threshold=corr_threshold),
        "atipicos": outliers_report(df),
        "calidad": data_quality_score(df),
    }


# --------------------------------------------------------------------------- #
# utilidades internas
# --------------------------------------------------------------------------- #
def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024.0  # type: ignore[assignment]
    return f"{n:.1f} PB"


def _safe(value: Any) -> Any:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 4)
    return value


def _first_value(serie: pd.Series) -> Any:
    notna = serie.dropna()
    return notna.iloc[0] if len(notna) else None


def _logical_kind(serie: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(serie):
        return "booleano"
    if pd.api.types.is_datetime64_any_dtype(serie):
        return "fecha"
    if pd.api.types.is_numeric_dtype(serie):
        return "numerico"
    if isinstance(serie.dtype, pd.CategoricalDtype):
        return "categorico"
    if serie.nunique(dropna=True) <= max(20, int(0.05 * max(len(serie), 1))):
        return "categorico"
    return "texto"


def _bounds(serie: pd.Series, method: OutlierMethod, threshold: float) -> tuple[Any, Any]:
    s = pd.to_numeric(serie, errors="coerce")
    if method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return q1 - threshold * iqr, q3 + threshold * iqr
    if method == "zscore":
        return s.mean() - threshold * s.std(), s.mean() + threshold * s.std()
    med = s.median()
    mad = (s - med).abs().median()
    return med - threshold * mad / 0.6745, med + threshold * mad / 0.6745
