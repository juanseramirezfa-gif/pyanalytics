"""Ingenieria de caracteristicas (feature engineering).

Incluye variables de fecha, interacciones, polinomios, rezagos/medias moviles,
agregados por grupo, texto y un preprocesador de scikit-learn listo para usar
(ColumnTransformer), siguiendo el enfoque de las fuentes del notebook.

Ejemplos
--------
>>> from pyanalytics.features import add_date_features, build_preprocessor
>>> df2 = add_date_features(df, "fecha")
>>> pre = build_preprocessor(df, target="precio")
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from .profile import categorical_columns, datetime_columns, numeric_columns, text_columns

__all__ = [
    "add_date_features",
    "add_interactions",
    "add_polynomial",
    "add_lag_features",
    "add_rolling_features",
    "add_aggregate_features",
    "add_ratio_features",
    "add_text_features",
    "add_frequency_features",
    "build_preprocessor",
    "select_by_variance",
    "select_by_correlation",
    "select_by_importance",
]


def add_date_features(
    df: pd.DataFrame,
    column: str,
    *,
    features: list[str] | None = None,
    drop_original: bool = False,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Extrae componentes de una columna de fecha/hora.

    Componentes disponibles: ``year``, ``month``, ``day``, ``dayofweek``,
    ``dayofyear``, ``week``, ``quarter``, ``hour``, ``minute``, ``is_weekend``,
    ``is_month_start``, ``is_month_end``, ``mes_sin``, ``mes_cos``,
    ``dia_semana_sin``, ``dia_semana_cos`` (codificacion ciclica).
    """
    out = df.copy()
    serie = out[column]
    if not pd.api.types.is_datetime64_any_dtype(serie):
        serie = pd.to_datetime(serie, errors="coerce", format="mixed")
        out[column] = serie

    dia_semana = _dt_accessor(serie, "day_of_week", "dayofweek")
    dia_anio = _dt_accessor(serie, "day_of_year", "dayofyear")

    disponibles: dict[str, pd.Series] = {
        "year": serie.dt.year,
        "month": serie.dt.month,
        "day": serie.dt.day,
        "dayofweek": dia_semana,
        "dayofyear": dia_anio,
        "week": serie.dt.isocalendar().week.astype("float"),
        "quarter": serie.dt.quarter,
        "hour": serie.dt.hour,
        "minute": serie.dt.minute,
        "is_weekend": (dia_semana >= 5).astype("float"),
        "is_month_start": serie.dt.is_month_start.astype("float"),
        "is_month_end": serie.dt.is_month_end.astype("float"),
        "mes_sin": np.sin(2 * np.pi * serie.dt.month / 12),
        "mes_cos": np.cos(2 * np.pi * serie.dt.month / 12),
        "dia_semana_sin": np.sin(2 * np.pi * dia_semana / 7),
        "dia_semana_cos": np.cos(2 * np.pi * dia_semana / 7),
    }
    seleccion = features or [
        "year",
        "month",
        "day",
        "dayofweek",
        "week",
        "quarter",
        "is_weekend",
        "mes_sin",
        "mes_cos",
    ]
    pref = prefix if prefix is not None else column
    for nombre in seleccion:
        if nombre not in disponibles:
            raise ValueError(f"Componente de fecha desconocido: {nombre!r}")
        out[f"{pref}_{nombre}"] = disponibles[nombre]
    if drop_original:
        out = out.drop(columns=[column])
    return out


def add_interactions(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    operation: Literal["product", "sum", "difference", "ratio"] = "product",
    max_pairs: int = 20,
) -> pd.DataFrame:
    """Crea combinaciones entre columnas numericas (producto, suma, resta, razon)."""
    cols = columns or numeric_columns(df)
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    out = df.copy()
    creadas = 0
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            if creadas >= max_pairs:
                return out
            if operation == "product":
                out[f"{a}_x_{b}"] = df[a] * df[b]
            elif operation == "sum":
                out[f"{a}_mas_{b}"] = df[a] + df[b]
            elif operation == "difference":
                out[f"{a}_menos_{b}"] = df[a] - df[b]
            elif operation == "ratio":
                out[f"{a}_div_{b}"] = df[a] / df[b].replace(0, np.nan)
            else:
                raise ValueError("operation debe ser product, sum, difference o ratio.")
            creadas += 1
    return out


def add_polynomial(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    degree: int = 2,
    *,
    include_interaction: bool = True,
) -> pd.DataFrame:
    """Anade potencias (y opcionalmente interacciones) de columnas numericas."""
    from sklearn.preprocessing import PolynomialFeatures

    cols = columns or numeric_columns(df)
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return df.copy()
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=not include_interaction)
    valores = poly.fit_transform(df[cols].astype(float))
    nombres = [n.replace(" ", "_").replace("^", "_pow") for n in poly.get_feature_names_out(cols)]
    nuevas = pd.DataFrame(valores, columns=nombres, index=df.index)
    nuevas = nuevas.drop(columns=[c for c in cols if c in nuevas.columns], errors="ignore")
    return pd.concat([df, nuevas], axis=1)


def add_lag_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    lags: list[int] | tuple[int, ...] = (1, 2, 3),
    *,
    sort_by: str | None = None,
) -> pd.DataFrame:
    """Crea rezagos (valores pasados) de una o varias columnas."""
    cols = columns or numeric_columns(df)
    out = df.sort_values(sort_by).copy() if sort_by else df.copy()
    for col in cols:
        for lag in lags:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    windows: list[int] | tuple[int, ...] = (3, 7),
    aggs: list[str] | tuple[str, ...] = ("mean", "std"),
    *,
    sort_by: str | None = None,
) -> pd.DataFrame:
    """Crea medias/desviaciones moviles (ventanas) para series temporales."""
    cols = columns or numeric_columns(df)
    out = df.sort_values(sort_by).copy() if sort_by else df.copy()
    for col in cols:
        for w in windows:
            roll = out[col].rolling(window=w, min_periods=1)
            for agg in aggs:
                out[f"{col}_roll{w}_{agg}"] = getattr(roll, agg)()
    return out


def add_aggregate_features(
    df: pd.DataFrame,
    group_column: str,
    columns: list[str] | None = None,
    aggs: list[str] | tuple[str, ...] = ("mean", "max", "min"),
    *,
    prefix: str = "grp",
) -> pd.DataFrame:
    """Agrega por grupo y pega el resultado a cada fila.

    Util para comparar cada registro con la media de su grupo (ciudad, tienda...).
    """
    cols = columns or numeric_columns(df)
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    out = df.copy()
    agrupado = out.groupby(group_column, dropna=False)[cols]
    for agg in aggs:
        resultado = agrupado.transform(agg)
        for col in cols:
            out[f"{prefix}_{col}_{agg}"] = resultado[col]
    return out


def add_ratio_features(
    df: pd.DataFrame,
    numerator: str,
    denominator: str,
    *,
    name: str | None = None,
) -> pd.DataFrame:
    """Crea una razon entre dos columnas evitando divisiones por cero."""
    out = df.copy()
    out[name or f"{numerator}_sobre_{denominator}"] = df[numerator] / df[denominator].replace(0, np.nan)
    return out


def add_text_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    drop_original: bool = False,
) -> pd.DataFrame:
    """Anade longitud, numero de palabras y de digitos de columnas de texto."""
    cols = columns or text_columns(df)
    out = df.copy()
    for col in cols:
        texto = out[col].astype("string")
        out[f"{col}_len"] = texto.str.len().astype("float")
        out[f"{col}_palabras"] = texto.str.split().str.len().astype("float")
        out[f"{col}_digitos"] = texto.str.count(r"\d").astype("float")
        if drop_original:
            out = out.drop(columns=[col])
    return out


def add_frequency_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Anade la frecuencia relativa de cada categoria (encoding por frecuencia)."""
    cols = columns or categorical_columns(df)
    out = df.copy()
    for col in cols:
        out[f"{col}_frecuencia"] = out[col].map(out[col].value_counts(normalize=True)).astype(float)
    return out


def build_preprocessor(
    df: pd.DataFrame,
    *,
    numeric_columns_: list[str] | None = None,
    categorical_columns_: list[str] | None = None,
    numeric_strategy: str = "median",
    scale_numeric: bool = True,
    scale_method: str = "zscore",
    categorical_strategy: Literal["onehot", "ordinal"] = "onehot",
    max_categories: int | None = 20,
    include_remainder: bool = False,
):
    """Construye un ``ColumnTransformer`` de scikit-learn para preprocesar datos.

    Devuelve un objeto listo para usar dentro de un ``Pipeline`` de scikit-learn:
    imputa nulos, escala numericas y codifica categoricas en un solo paso.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler

    num = numeric_columns_ if numeric_columns_ is not None else numeric_columns(df)
    cat = categorical_columns_ if categorical_columns_ is not None else categorical_columns(df)
    if max_categories:
        cat = [c for c in cat if df[c].nunique(dropna=True) <= max_categories]

    pasos_num: list[tuple[str, Any]] = [("imputar", SimpleImputer(strategy=numeric_strategy))]
    if scale_numeric:
        escaladores = {
            "zscore": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
        }
        pasos_num.append(("escalar", escaladores.get(scale_method, StandardScaler)()))
    transformador_num = Pipeline(pasos_num)

    if categorical_strategy == "onehot":
        codificador = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=None)
    else:
        codificador = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    transformador_cat = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", codificador),
        ]
    )

    return ColumnTransformer(
        [
            ("numericas", transformador_num, num),
            ("categoricas", transformador_cat, cat),
        ],
        remainder="passthrough" if include_remainder else "drop",
    )


def select_by_variance(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = 0.0,
) -> list[str]:
    """Devuelve las columnas numericas con varianza por encima del umbral."""
    from sklearn.feature_selection import VarianceThreshold

    cols = columns or numeric_columns(df)
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return []
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df[cols].fillna(df[cols].median()))
    return [c for c, keep in zip(cols, selector.get_support()) if keep]


def select_by_correlation(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = 0.9,
) -> list[str]:
    """Elimina columnas redundantes muy correlacionadas entre si."""
    cols = columns or numeric_columns(df)
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) < 2:
        return cols
    corr = df[cols].corr().abs()
    superior = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    redundantes = [c for c in superior.columns if (superior[c] > threshold).any()]
    return [c for c in cols if c not in redundantes]


def select_by_importance(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    k: int = 10,
    task: Literal["auto", "classification", "regression"] = "auto",
    random_state: int = 0,
) -> pd.DataFrame:
    """Ranking de caracteristicas por importancia (Random Forest / permutacion)."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer

    if task == "auto":
        task = "classification" if y.nunique() <= 20 and not pd.api.types.is_float_dtype(y) else "regression"
    modelo = (
        RandomForestClassifier(n_estimators=200, random_state=random_state)
        if task == "classification"
        else RandomForestRegressor(n_estimators=200, random_state=random_state)
    )
    numerico = X.select_dtypes(include=[np.number]).copy()
    imputado = SimpleImputer(strategy="median").fit_transform(numerico)
    modelo.fit(imputado, y)
    tabla = pd.DataFrame(
        {"caracteristica": numerico.columns, "importancia": modelo.feature_importances_}
    ).sort_values("importancia", ascending=False, ignore_index=True)
    tabla["acumulada"] = tabla["importancia"].cumsum().round(4)
    tabla["seleccionada"] = tabla.index < k
    return tabla


def _dt_accessor(serie: pd.Series, nuevo: str, antiguo: str) -> pd.Series:
    """Accede a un atributo de ``.dt`` con compatibilidad pandas 2 y 3.

    pandas 3 renombro ``dayofweek`` -> ``day_of_week`` y ``dayofyear`` ->
    ``day_of_year``. Se prefiere el nombre nuevo si existe.
    """
    accesor = serie.dt
    if hasattr(accesor, nuevo):
        return getattr(accesor, nuevo)
    return getattr(accesor, antiguo)
