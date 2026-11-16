"""Limpieza de datos: nulos, duplicados, tipos, texto y atipicos.

Cada funcion devuelve un DataFrame nuevo (no modifica el original) y puede
registrar lo que hizo en un :class:`CleanReport` para dejar trazabilidad.

Ejemplos
--------
>>> from pyanalytics.clean import auto_clean
>>> limpio, informe = auto_clean(df)
>>> informe.summary()
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from .profile import detect_outliers, numeric_columns, text_columns

__all__ = [
    "CleanReport",
    "normalize_column_names",
    "strip_whitespace",
    "drop_empty_columns",
    "drop_constant_columns",
    "drop_duplicates",
    "drop_missing_rows",
    "impute_missing",
    "convert_types",
    "smart_cast",
    "replace_values",
    "cap_outliers",
    "remove_outliers",
    "auto_clean",
    "clean_pipeline",
]

MissingStrategy = Literal["auto", "drop", "mean", "median", "mode", "constant", "ffill", "bfill", "interpolate", "zero"]
OutlierMethod = Literal["iqr", "zscore", "mad"]


@dataclass
class CleanReport:
    """Registro de las transformaciones aplicadas a un DataFrame."""

    pasos: list[dict[str, Any]] = field(default_factory=list)

    def add(self, accion: str, detalle: str, filas: int = 0, columnas: int = 0, **extra: Any) -> None:
        self.pasos.append(
            {"accion": accion, "detalle": detalle, "filas_afectadas": filas, "columnas_afectadas": columnas, **extra}
        )

    def to_frame(self) -> pd.DataFrame:
        if not self.pasos:
            return pd.DataFrame(columns=["accion", "detalle", "filas_afectadas", "columnas_afectadas"])
        return pd.DataFrame(self.pasos)

    def summary(self) -> str:
        if not self.pasos:
            return "Sin transformaciones registradas."
        lineas = [f"- {p['accion']}: {p['detalle']}" for p in self.pasos]
        return "\n".join(lineas)

    def __len__(self) -> int:
        return len(self.pasos)

    def __repr__(self) -> str:  # pragma: no cover
        return f"CleanReport({len(self.pasos)} pasos)"


def _rec(report: CleanReport | None, *args: Any, **kwargs: Any) -> None:
    if report is not None:
        report.add(*args, **kwargs)


def _snake(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"[^\w\s]", "_", s, flags=re.UNICODE)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_").lower() or "columna"


def normalize_column_names(
    df: pd.DataFrame,
    case: Literal["snake", "lower", "upper", "keep"] = "snake",
    report: CleanReport | None = None,
) -> pd.DataFrame:
    """Estandariza los nombres de columna (sin espacios ni simbolos raros)."""
    original = list(df.columns)
    if case == "snake":
        nuevos = [_snake(c) for c in original]
    elif case == "lower":
        nuevos = [str(c).strip().lower() for c in original]
    elif case == "upper":
        nuevos = [str(c).strip().upper() for c in original]
    else:
        nuevos = [str(c).strip() for c in original]

    # garantiza unicidad
    vistos: dict[str, int] = {}
    unicos: list[str] = []
    for name in nuevos:
        if name in vistos:
            vistos[name] += 1
            unicos.append(f"{name}_{vistos[name]}")
        else:
            vistos[name] = 0
            unicos.append(name)

    cambiadas = sum(1 for a, b in zip(original, unicos) if str(a) != b)
    out = df.copy()
    out.columns = unicos
    _rec(report, "nombres_columnas", f"{cambiadas} nombre(s) normalizados a formato {case}", columnas=cambiadas)
    return out


def strip_whitespace(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    report: CleanReport | None = None,
) -> pd.DataFrame:
    """Elimina espacios sobrantes al inicio/fin en columnas de texto."""
    out = df.copy()
    cols = columns or text_columns(out)
    afectadas = 0
    for col in cols:
        if not (pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col])):
            continue
        antes = out[col]
        despues = antes.map(lambda v: v.strip() if isinstance(v, str) else v)
        a, d = antes.astype("object"), despues.astype("object")
        cambios = int(((a != d) & ~(a.isna() & d.isna())).sum())
        out[col] = despues
        afectadas += cambios
    _rec(report, "espacios", f"{afectadas} valor(es) con espacios recortados", filas=afectadas, columnas=len(cols))
    return out


def drop_empty_columns(
    df: pd.DataFrame,
    threshold: float = 1.0,
    report: CleanReport | None = None,
) -> pd.DataFrame:
    """Elimina columnas cuyo porcentaje de nulos supera ``threshold`` (0-1)."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold debe estar entre 0 y 1.")
    ratios = df.isna().mean()
    eliminar = ratios[ratios >= threshold].index.tolist()
    out = df.drop(columns=eliminar)
    _rec(report, "columnas_vacias", f"eliminadas {len(eliminar)}: {', '.join(map(str, eliminar[:8]))}", columnas=len(eliminar))
    return out


def drop_constant_columns(
    df: pd.DataFrame,
    report: CleanReport | None = None,
) -> pd.DataFrame:
    """Elimina columnas con un unico valor (no aportan informacion)."""
    eliminar = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    out = df.drop(columns=eliminar)
    _rec(report, "columnas_constantes", f"eliminadas {len(eliminar)}: {', '.join(map(str, eliminar[:8]))}", columnas=len(eliminar))
    return out


def drop_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    keep: Literal["first", "last", False] = "first",
    report: CleanReport | None = None,
) -> pd.DataFrame:
    """Elimina filas duplicadas."""
    antes = len(df)
    out = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    _rec(report, "duplicados", f"eliminadas {antes - len(out)} fila(s) duplicada(s)", filas=antes - len(out))
    return out


def drop_missing_rows(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    how: Literal["any", "all"] = "any",
    thresh: int | None = None,
    report: CleanReport | None = None,
) -> pd.DataFrame:
    """Elimina filas con valores nulos."""
    antes = len(df)
    out = df.dropna(subset=subset, how=how, thresh=thresh).reset_index(drop=True)
    _rec(report, "filas_nulas", f"eliminadas {antes - len(out)} fila(s) con nulos", filas=antes - len(out))
    return out


def impute_missing(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    strategy: MissingStrategy = "auto",
    fill_value: Any = None,
    group_by: str | None = None,
    report: CleanReport | None = None,
) -> pd.DataFrame:
    """Rellena valores nulos.

    ``strategy``:
      * ``"auto"``   media/mediana segun asimetria para numericas, moda para el resto
      * ``"drop"``   elimina las filas con nulos
      * ``"mean"``, ``"median"``, ``"mode"``, ``"zero"``, ``"constant"``
      * ``"ffill"``, ``"bfill"``, ``"interpolate"`` (series temporales)

    ``group_by`` permite imputar por grupo (p. ej. la mediana por ciudad).
    """
    if strategy == "drop":
        return drop_missing_rows(df, subset=columns, report=report)

    out = df.copy()
    cols = columns or out.columns.tolist()
    detalles: list[str] = []

    for col in cols:
        serie = out[col]
        if not serie.isna().any():
            continue
        es_num = pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie)
        estrategia = strategy
        if estrategia == "auto":
            if es_num:
                skew = float(serie.skew()) if serie.notna().sum() > 2 else 0.0
                estrategia = "median" if abs(skew) > 1 else "mean"
            elif pd.api.types.is_datetime64_any_dtype(serie):
                estrategia = "ffill"
            else:
                estrategia = "mode"

        if not es_num and estrategia in ("mean", "median"):
            estrategia = "mode"
        if not es_num and estrategia == "interpolate":
            estrategia = "ffill"

        if group_by and group_by in out.columns and es_num and estrategia in ("mean", "median"):
            agg = "mean" if estrategia == "mean" else "median"
            relleno = out.groupby(group_by, dropna=False)[col].transform(agg)
            out[col] = serie.fillna(relleno)
            out[col] = out[col].fillna(out[col].median() if agg == "median" else out[col].mean())
        elif estrategia in ("mean", "median"):
            valor = serie.mean() if estrategia == "mean" else serie.median()
            out[col] = serie.fillna(valor)
        elif estrategia == "mode":
            modo = serie.mode(dropna=True)
            out[col] = serie.fillna(modo.iloc[0] if len(modo) else "desconocido")
        elif estrategia == "zero":
            out[col] = serie.fillna(0)
        elif estrategia == "constant":
            if fill_value is None:
                raise ValueError("Con strategy='constant' debes indicar fill_value.")
            out[col] = serie.fillna(fill_value)
        elif estrategia == "ffill":
            out[col] = serie.ffill().bfill()
        elif estrategia == "bfill":
            out[col] = serie.bfill().ffill()
        elif estrategia == "interpolate":
            out[col] = serie.interpolate(method="linear", limit_direction="both")
        else:
            raise ValueError(f"Estrategia desconocida: {estrategia}")

        n = int(serie.isna().sum())
        detalles.append(f"{col}({estrategia},{n})")

    _rec(report, "nulos_rellenados", "; ".join(detalles) or "nada que rellenar", columnas=len(detalles))
    return out


def smart_cast(serie: pd.Series) -> pd.Series:
    """Convierte texto 'sucio' a numero o fecha cuando es posible.

    Reconoce separadores de miles/decimales, simbolos de moneda y porcentajes.
    """
    if not (pd.api.types.is_object_dtype(serie) or pd.api.types.is_string_dtype(serie)):
        return serie

    muestra = serie.dropna().astype(str).head(500)
    if muestra.empty:
        return serie

    limpio = muestra.str.strip()
    es_pct = bool(limpio.str.endswith("%").mean() > 0.6)
    sin_simbolos = (
        limpio.str.replace(r"[^\d,.\-+eE]", "", regex=True)
        .str.replace(r"^\+", "", regex=True)
    )

    def _a_numero(texto: str) -> float | None:
        if texto in ("", "-", ".", ",", "-.", "-,"):
            return None
        tiene_coma, tiene_punto = "," in texto, "." in texto
        if tiene_coma and tiene_punto:
            if texto.rfind(",") > texto.rfind("."):
                texto = texto.replace(".", "").replace(",", ".")
            else:
                texto = texto.replace(",", "")
        elif tiene_coma:
            partes = texto.split(",")
            texto = texto.replace(",", ".") if len(partes[-1]) != 3 else texto.replace(",", "")
        try:
            return float(texto)
        except ValueError:
            return None

    convertidos = sin_simbolos.map(_a_numero)
    if convertidos.notna().mean() >= 0.9:
        resultado = pd.to_numeric(sin_simbolos.map(_a_numero), errors="coerce")
        if es_pct:
            resultado = resultado / 100.0
        return resultado.reindex(serie.index)

    fechas = pd.to_datetime(serie, errors="coerce", format="mixed")
    if fechas.notna().mean() >= 0.9 and fechas.notna().sum() > 0:
        return fechas

    return serie


def convert_types(
    df: pd.DataFrame,
    mapping: dict[str, str] | None = None,
    auto: bool = True,
    report: CleanReport | None = None,
) -> pd.DataFrame:
    """Convierte tipos de columna.

    ``mapping`` fuerza tipos, por ejemplo ``{"fecha": "datetime64[ns]", "precio": "float64"}``.
    Con ``auto=True`` se detectan numeros y fechas escritos como texto.
    """
    out = df.copy()
    cambios: list[str] = []

    if auto:
        for col in text_columns(out):
            antes = out[col].dtype
            convertida = smart_cast(out[col])
            if convertida.dtype != antes:
                out[col] = convertida
                cambios.append(f"{col}:{antes}->{convertida.dtype}")

    if mapping:
        for col, tipo in mapping.items():
            if col not in out.columns:
                raise KeyError(f"La columna {col!r} no existe.")
            antes = out[col].dtype
            if tipo.startswith("datetime"):
                out[col] = pd.to_datetime(out[col], errors="coerce", format="mixed")
            elif tipo in ("category", "categoria"):
                out[col] = out[col].astype("category")
            else:
                out[col] = out[col].astype(tipo)
            cambios.append(f"{col}:{antes}->{out[col].dtype}")

    _rec(report, "tipos", "; ".join(cambios) or "sin cambios de tipo", columnas=len(cambios))
    return out


def replace_values(
    df: pd.DataFrame,
    column: str,
    mapping: dict[Any, Any],
    report: CleanReport | None = None,
) -> pd.DataFrame:
    """Reemplaza valores concretos de una columna."""
    out = df.copy()
    antes = out[column]
    out[column] = antes.replace(mapping)
    n = int((antes != out[column]).sum())
    _rec(report, "reemplazos", f"{column}: {n} valor(es) reemplazados", filas=n, columnas=1)
    return out


def cap_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: OutlierMethod = "iqr",
    threshold: float = 1.5,
    report: CleanReport | None = None,
) -> pd.DataFrame:
    """Recorta (winsoriza) los atipicos a los limites calculados."""
    out = df.copy()
    cols = columns or numeric_columns(out)
    detalles: list[str] = []
    for col in cols:
        serie = pd.to_numeric(out[col], errors="coerce")
        if serie.notna().sum() < 3:
            continue
        if method == "iqr":
            q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - threshold * iqr, q3 + threshold * iqr
        elif method == "zscore":
            low, high = serie.mean() - threshold * serie.std(), serie.mean() + threshold * serie.std()
        else:
            med = serie.median()
            mad = (serie - med).abs().median()
            low, high = med - threshold * mad / 0.6745, med + threshold * mad / 0.6745
        n = int(((serie < low) | (serie > high)).sum())
        if n:
            out[col] = serie.clip(lower=low, upper=high)
            detalles.append(f"{col}({n})")
    _rec(report, "atipicos_recortados", "; ".join(detalles) or "nada que recortar", columnas=len(detalles))
    return out


def remove_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: OutlierMethod = "iqr",
    threshold: float = 1.5,
    report: CleanReport | None = None,
) -> pd.DataFrame:
    """Elimina las filas que tienen atipicos en las columnas indicadas."""
    out = df.copy()
    cols = columns or numeric_columns(out)
    mascara = pd.Series(False, index=out.index)
    for col in cols:
        mascara |= detect_outliers(out, col, method=method, threshold=threshold)
    resultado = out[~mascara].reset_index(drop=True)
    _rec(report, "atipicos_eliminados", f"eliminadas {int(mascara.sum())} fila(s)", filas=int(mascara.sum()), columnas=len(cols))
    return resultado


def auto_clean(
    df: pd.DataFrame,
    missing_threshold: float = 0.6,
    duplicate_subset: list[str] | None = None,
    cast_types: bool = True,
) -> tuple[pd.DataFrame, CleanReport]:
    """Limpieza razonable por defecto, paso a paso y con trazabilidad.

    1. Normaliza nombres de columna.
    2. Recorta espacios en texto.
    3. Convierte tipos mal tipados (numeros/fechas como texto).
    4. Elimina columnas vacias y constantes.
    5. Elimina duplicados.
    6. Imputa nulos de forma automatica.
    """
    report = CleanReport()
    out = normalize_column_names(df, report=report)
    out = strip_whitespace(out, report=report)
    if cast_types:
        out = convert_types(out, auto=True, report=report)
    out = drop_empty_columns(out, threshold=missing_threshold, report=report)
    out = drop_constant_columns(out, report=report)
    out = drop_duplicates(out, subset=duplicate_subset, report=report)
    out = impute_missing(out, strategy="auto", report=report)
    return out, report


def clean_pipeline(
    df: pd.DataFrame,
    steps: list[dict[str, Any]],
) -> tuple[pd.DataFrame, CleanReport]:
    """Ejecuta una secuencia de limpieza declarativa.

    Cada paso es ``{"funcion": <nombre o callable>, **argumentos}``.

    >>> pasos = [
    ...     {"funcion": "normalize_column_names"},
    ...     {"funcion": "drop_duplicates"},
    ...     {"funcion": "impute_missing", "strategy": "median"},
    ... ]
    >>> limpio, informe = clean_pipeline(df, pasos)
    """
    registro: dict[str, Callable[..., pd.DataFrame]] = {
        "normalize_column_names": normalize_column_names,
        "strip_whitespace": strip_whitespace,
        "drop_empty_columns": drop_empty_columns,
        "drop_constant_columns": drop_constant_columns,
        "drop_duplicates": drop_duplicates,
        "drop_missing_rows": drop_missing_rows,
        "impute_missing": impute_missing,
        "convert_types": convert_types,
        "replace_values": replace_values,
        "cap_outliers": cap_outliers,
        "remove_outliers": remove_outliers,
    }
    report = CleanReport()
    out = df.copy()
    for i, step in enumerate(steps, start=1):
        paso = dict(step)
        nombre = paso.pop("funcion", None)
        if nombre is None:
            raise ValueError(f"El paso {i} no define 'funcion'.")
        fn = registro.get(nombre) if isinstance(nombre, str) else nombre
        if fn is None:
            raise ValueError(f"Funcion desconocida en el paso {i}: {nombre!r}")
        paso.pop("report", None)
        out = fn(out, report=report, **paso)
    return out, report
