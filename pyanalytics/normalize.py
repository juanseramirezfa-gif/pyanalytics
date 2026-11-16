"""Normalizacion, escalado, codificacion y discretizacion.

Escalas disponibles
-------------------
``minmax``, ``zscore`` (estandar), ``robust``, ``maxabs``, ``decimal``,
``log``, ``log1p``, ``sqrt``, ``boxcox``, ``yeojohnson``, ``quantile``,
``rank``, ``l2``, ``binary``.

Ejemplos
--------
>>> from pyanalytics.normalize import normalize, Normalizer
>>> df_esc = normalize(df, ["edad", "salario"], method="minmax")
>>> esc = Normalizer("robust", ["salario"]).fit(train)
>>> test_esc = esc.transform(test)
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    Binarizer,
    KBinsDiscretizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer as SkNormalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from .profile import categorical_columns, numeric_columns

__all__ = [
    "SCALE_METHODS",
    "normalize",
    "min_max",
    "z_score",
    "robust_scale",
    "max_abs",
    "decimal_scaling",
    "log_transform",
    "boxcox_transform",
    "quantile_transform",
    "rank_transform",
    "Normalizer",
    "encode_categorical",
    "discretize",
    "recommend_method",
    "split_scale",
]

SCALE_METHODS = (
    "minmax",
    "zscore",
    "robust",
    "maxabs",
    "decimal",
    "log",
    "log1p",
    "sqrt",
    "boxcox",
    "yeojohnson",
    "quantile",
    "rank",
    "l2",
    "binary",
)

_SKLEARN_SCALERS: dict[str, type] = {
    "minmax": MinMaxScaler,
    "zscore": StandardScaler,
    "robust": RobustScaler,
    "maxabs": MaxAbsScaler,
    "quantile": QuantileTransformer,
    "l2": SkNormalizer,
    "binary": Binarizer,
    "yeojohnson": PowerTransformer,
}


# --------------------------------------------------------------------------- #
# funciones sueltas
# --------------------------------------------------------------------------- #
def min_max(df: pd.DataFrame, columns: list[str] | None = None, feature_range: tuple[float, float] = (0.0, 1.0)) -> pd.DataFrame:
    """Escala a un rango fijo (por defecto 0-1)."""
    return normalize(df, columns, method="minmax", feature_range=feature_range)


def z_score(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Estandariza a media 0 y desviacion 1."""
    return normalize(df, columns, method="zscore")


def robust_scale(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Escala con mediana y rango intercuartilico (resistente a atipicos)."""
    return normalize(df, columns, method="robust")


def max_abs(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Divide por el maximo valor absoluto (conserva los ceros)."""
    return normalize(df, columns, method="maxabs")


def decimal_scaling(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Escalado decimal: divide por 10**j donde j es el orden de magnitud maximo."""
    return normalize(df, columns, method="decimal")


def log_transform(df: pd.DataFrame, columns: list[str] | None = None, base: float | None = None) -> pd.DataFrame:
    """Transformacion logaritmica (desplaza si hay ceros o negativos)."""
    return normalize(df, columns, method="log", base=base)


def boxcox_transform(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Transformacion Box-Cox (requiere valores estrictamente positivos)."""
    return normalize(df, columns, method="boxcox")


def quantile_transform(df: pd.DataFrame, columns: list[str] | None = None, n_quantiles: int = 1000) -> pd.DataFrame:
    """Transforma a una distribucion uniforme/normal por cuantiles."""
    return normalize(df, columns, method="quantile", n_quantiles=n_quantiles)


def rank_transform(df: pd.DataFrame, columns: list[str] | None = None, method: str = "average") -> pd.DataFrame:
    """Convierte a rangos (robusto ante valores extremos)."""
    return normalize(df, columns, method="rank", rank_method=method)


# --------------------------------------------------------------------------- #
# funcion principal
# --------------------------------------------------------------------------- #
def normalize(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "zscore",
    *,
    suffix: str | None = None,
    keep_original: bool = True,
    report: Any = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Escala/normaliza columnas numericas.

    Parametros
    ----------
    columns:
        Columnas a escalar. Por defecto, todas las numericas.
    method:
        Uno de :data:`SCALE_METHODS`.
    suffix:
        Sufijo para las nuevas columnas (por defecto ``_<method>``).
    keep_original:
        Si es ``True`` conserva las columnas originales y anade las escaladas.
    **kwargs:
        Se pasan al escalador (``feature_range``, ``n_quantiles``, ``base``...).
    """
    if method not in SCALE_METHODS:
        raise ValueError(f"Metodo no soportado: {method!r}. Usa uno de {SCALE_METHODS}.")
    cols = columns or numeric_columns(df)
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return df.copy()

    out = df.copy()
    sufijo = suffix if suffix is not None else f"_{method}"
    datos = out[cols].astype(float)

    if method == "decimal":
        escalado = datos.copy()
        for col in cols:
            max_abs_val = np.nanmax(np.abs(datos[col].to_numpy())) if datos[col].notna().any() else 0.0
            j = 0 if max_abs_val == 0 else int(np.ceil(np.log10(max_abs_val)))
            escalado[col] = datos[col] / (10.0**j)
    elif method == "log":
        base = kwargs.get("base")
        escalado = _safe_log(datos, base=base)
    elif method == "log1p":
        escalado = _safe_log(datos, base=kwargs.get("base"), offset=1.0)
    elif method == "sqrt":
        escalado = _safe_sqrt(datos)
    elif method == "boxcox":
        escalado = _boxcox_frame(datos)
    elif method == "rank":
        escalado = datos.rank(method=kwargs.get("rank_method", "average"), pct=kwargs.get("pct", False))
    else:
        escalado = _fit_sklearn(method, datos, **kwargs)

    escalado.index = out.index
    if keep_original:
        for col in cols:
            out[f"{col}{sufijo}"] = escalado[col]
    else:
        out[cols] = escalado[cols]
    return out


def _fit_sklearn(method: str, datos: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    if method == "quantile":
        params = {"n_quantiles": kwargs.pop("n_quantiles", min(1000, max(len(datos), 2))), "output_distribution": kwargs.pop("output_distribution", "uniform"), "subsample": kwargs.pop("subsample", 100000), "random_state": kwargs.pop("random_state", 0)}
        scaler = QuantileTransformer(**params)
    elif method == "l2":
        scaler = SkNormalizer(norm=kwargs.pop("norm", "l2"))
    elif method == "binary":
        scaler = Binarizer(threshold=kwargs.pop("threshold", 0.0))
    elif method == "yeojohnson":
        scaler = PowerTransformer(method="yeo-johnson", standardize=kwargs.pop("standardize", True))
    else:
        scaler = _SKLEARN_SCALERS[method](**kwargs)
    valores = scaler.fit_transform(datos)
    return pd.DataFrame(valores, columns=datos.columns, index=datos.index)


def _safe_log(datos: pd.DataFrame, base: float | None = None, offset: float = 0.0) -> pd.DataFrame:
    out = datos.copy()
    for col in datos.columns:
        serie = datos[col]
        min_val = float(serie.min(skipna=True)) if serie.notna().any() else 0.0
        shift = offset
        if min_val + shift <= 0:
            shift = -min_val + 1.0 + offset
        valores = np.log(serie + shift)
        if base:
            valores = valores / np.log(base)
        out[col] = valores
    return out


def _safe_sqrt(datos: pd.DataFrame) -> pd.DataFrame:
    out = datos.copy()
    for col in datos.columns:
        serie = datos[col]
        min_val = float(serie.min(skipna=True)) if serie.notna().any() else 0.0
        out[col] = np.sqrt(serie - min_val) if min_val < 0 else np.sqrt(serie)
    return out


def _boxcox_frame(datos: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats

    out = datos.copy()
    for col in datos.columns:
        serie = datos[col]
        if serie.notna().any() and float(serie.min(skipna=True)) > 0:
            validos = serie.dropna()
            transformada, _ = stats.boxcox(validos)
            out.loc[validos.index, col] = transformada
        else:
            out[col] = _safe_log(datos[[col]], offset=0.0)[col]
    return out


# --------------------------------------------------------------------------- #
# clase reutilizable (fit en train, transform en test)
# --------------------------------------------------------------------------- #
class Normalizer:
    """Escalador con estado: se ajusta en entrenamiento y se reutiliza.

    >>> esc = Normalizer("minmax", ["edad", "salario"]).fit(train)
    >>> train_esc = esc.transform(train)
    >>> test_esc = esc.transform(test)
    """

    def __init__(self, method: str = "zscore", columns: list[str] | None = None, **kwargs: Any) -> None:
        if method not in SCALE_METHODS:
            raise ValueError(f"Metodo no soportado: {method!r}.")
        self.method = method
        self.columns = columns
        self.kwargs = kwargs
        self._cols: list[str] = []
        self._scaler: Any = None
        self._decimal_pow: dict[str, int] = {}
        self._log_shift: dict[str, float] = {}
        self._sqrt_shift: dict[str, float] = {}
        self._boxcox_lambda: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "Normalizer":
        cols = self.columns or numeric_columns(df)
        self._cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        datos = df[self._cols].astype(float)

        if self.method == "decimal":
            for col in self._cols:
                max_abs_val = np.nanmax(np.abs(datos[col].to_numpy())) if datos[col].notna().any() else 0.0
                self._decimal_pow[col] = 0 if max_abs_val == 0 else int(np.ceil(np.log10(max_abs_val)))
        elif self.method in ("log", "log1p"):
            offset = 1.0 if self.method == "log1p" else 0.0
            for col in self._cols:
                min_val = float(datos[col].min(skipna=True)) if datos[col].notna().any() else 0.0
                self._log_shift[col] = (-min_val + 1.0 + offset) if min_val + offset <= 0 else offset
        elif self.method == "sqrt":
            for col in self._cols:
                min_val = float(datos[col].min(skipna=True)) if datos[col].notna().any() else 0.0
                self._sqrt_shift[col] = -min_val if min_val < 0 else 0.0
        elif self.method == "boxcox":
            from scipy import stats

            for col in self._cols:
                serie = datos[col].dropna()
                if len(serie) > 2 and float(serie.min()) > 0:
                    _, lam = stats.boxcox(serie)
                    self._boxcox_lambda[col] = float(lam)
                else:
                    self._boxcox_lambda[col] = float("nan")
        elif self.method == "rank":
            pass
        else:
            self._scaler = _make_scaler(self.method, datos, self.kwargs)
            self._scaler.fit(datos)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._cols:
            raise RuntimeError("Llama a fit() antes de transform().")
        faltantes = [c for c in self._cols if c not in df.columns]
        if faltantes:
            raise KeyError(f"Faltan columnas en transform: {faltantes}")
        out = df.copy()
        datos = df[self._cols].astype(float)

        if self.method == "decimal":
            for col in self._cols:
                out[col] = datos[col] / (10.0 ** self._decimal_pow[col])
        elif self.method in ("log", "log1p"):
            base = self.kwargs.get("base")
            for col in self._cols:
                valores = np.log(datos[col] + self._log_shift[col])
                out[col] = valores / np.log(base) if base else valores
        elif self.method == "sqrt":
            for col in self._cols:
                out[col] = np.sqrt(datos[col] + self._sqrt_shift[col])
        elif self.method == "boxcox":
            from scipy import stats

            for col in self._cols:
                lam = self._boxcox_lambda[col]
                serie = datos[col]
                if np.isnan(lam):
                    out[col] = np.log(serie.clip(lower=1e-9))
                elif abs(lam) < 1e-9:
                    out[col] = np.log(serie)
                else:
                    out[col] = (serie**lam - 1.0) / lam
        elif self.method == "rank":
            out[self._cols] = datos.rank(method=self.kwargs.get("rank_method", "average"))
        else:
            valores = self._scaler.transform(datos)
            out[self._cols] = valores
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deshace la transformacion (no disponible para rank ni boxcox con lambda NaN)."""
        if self.method in ("rank",):
            raise NotImplementedError("La transformacion por rangos no es invertible.")
        out = df.copy()
        datos = df[self._cols].astype(float)
        if self.method == "decimal":
            for col in self._cols:
                out[col] = datos[col] * (10.0 ** self._decimal_pow[col])
        elif self.method in ("log", "log1p"):
            base = self.kwargs.get("base")
            for col in self._cols:
                valores = datos[col] * np.log(base) if base else datos[col]
                out[col] = np.exp(valores) - self._log_shift[col]
        elif self.method == "sqrt":
            for col in self._cols:
                out[col] = np.square(datos[col]) - self._sqrt_shift[col]
        else:
            out[self._cols] = self._scaler.inverse_transform(datos)
        return out


def _make_scaler(method: str, datos: pd.DataFrame, kwargs: dict[str, Any]) -> Any:
    params = dict(kwargs)
    if method == "quantile":
        params.setdefault("n_quantiles", min(1000, max(len(datos), 2)))
        params.setdefault("output_distribution", "uniform")
        params.setdefault("subsample", 100000)
        params.setdefault("random_state", 0)
    if method == "l2":
        params.pop("norm", None)
        return SkNormalizer(norm=kwargs.get("norm", "l2"))
    if method == "binary":
        return Binarizer(threshold=params.pop("threshold", 0.0))
    if method == "yeojohnson":
        return PowerTransformer(method="yeo-johnson", standardize=params.pop("standardize", True))
    cls = _SKLEARN_SCALERS[method]
    return cls(**params)


# --------------------------------------------------------------------------- #
# variables categoricas
# --------------------------------------------------------------------------- #
def encode_categorical(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: Literal["onehot", "ordinal", "label", "frequency", "target"] = "onehot",
    *,
    target: str | None = None,
    drop_first: bool = False,
    max_categories: int | None = None,
    keep_original: bool = False,
) -> pd.DataFrame:
    """Codifica variables categoricas a numerico.

    * ``onehot``    una columna binaria por categoria
    * ``ordinal``   un entero por categoria (orden de aparicion o alfabetico)
    * ``label``     entero por categoria (una sola columna por variable)
    * ``frequency`` frecuencia relativa de cada categoria
    * ``target``    media del target por categoria (requiere ``target=``)
    """
    cols = columns or categorical_columns(df)
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df.copy()
    out = df.copy()

    if method == "onehot":
        for col in cols:
            serie = out[col].astype("object").fillna("__nulo__")
            if max_categories and serie.nunique() > max_categories:
                top = serie.value_counts().head(max_categories).index
                serie = serie.where(serie.isin(top), "__otras__")
            dummies = pd.get_dummies(serie, prefix=col, drop_first=drop_first, dtype=int)
            out = pd.concat([out, dummies], axis=1)
            if not keep_original:
                out = out.drop(columns=[col])
    elif method in ("ordinal", "label"):
        for col in cols:
            serie = out[col].astype("object").fillna("__nulo__")
            categorias = sorted(serie.unique(), key=str)
            mapa = {cat: i for i, cat in enumerate(categorias)}
            out[f"{col}_cod"] = serie.map(mapa).astype(int)
            if not keep_original:
                out = out.drop(columns=[col])
    elif method == "frequency":
        for col in cols:
            frec = out[col].value_counts(normalize=True, dropna=False)
            out[f"{col}_frec"] = out[col].map(frec).astype(float)
            if not keep_original:
                out = out.drop(columns=[col])
    elif method == "target":
        if target is None or target not in out.columns:
            raise ValueError("Con method='target' debes indicar una columna 'target' existente.")
        out[target] = pd.to_numeric(out[target], errors="coerce")
        global_mean = float(out[target].mean())
        for col in cols:
            medias = out.groupby(col, dropna=False)[target].transform("mean")
            out[f"{col}_target"] = medias.fillna(global_mean).astype(float)
            if not keep_original:
                out = out.drop(columns=[col])
    else:
        raise ValueError(f"Metodo de codificacion no soportado: {method!r}")
    return out


def discretize(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    bins: int = 5,
    strategy: Literal["uniform", "quantile", "kmeans"] = "quantile",
    labels: list[str] | None = None,
    keep_original: bool = True,
) -> pd.DataFrame:
    """Convierte numericas continuas en intervalos (discretizacion)."""
    cols = columns or numeric_columns(df)
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return df.copy()
    out = df.copy()
    for col in cols:
        serie = df[col]
        if labels is not None:
            resultado = pd.cut(serie, bins=bins, labels=labels)
        elif strategy == "uniform":
            resultado = pd.cut(serie, bins=bins)
        elif strategy == "quantile":
            try:
                resultado = pd.qcut(serie, q=bins, duplicates="drop")
            except ValueError:
                resultado = pd.cut(serie, bins=bins)
        elif strategy == "kmeans":
            kb = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="kmeans", subsample=None)
            validos = serie.notna()
            etiquetas = pd.Series(index=serie.index, dtype="float")
            if validos.sum() > bins:
                etiquetas.loc[validos] = kb.fit_transform(serie[validos].to_numpy().reshape(-1, 1)).ravel()
            resultado = etiquetas
        else:
            raise ValueError("strategy debe ser 'uniform', 'quantile' o 'kmeans'.")
        if keep_original:
            out[f"{col}_bin"] = resultado.astype("object") if hasattr(resultado, "astype") else resultado
        else:
            out[col] = resultado
    return out


def recommend_method(serie: pd.Series) -> str:
    """Sugiere una escala segun la forma de la distribucion."""
    s = pd.to_numeric(serie, errors="coerce").dropna()
    if s.empty:
        return "zscore"
    asimetria = float(s.skew()) if len(s) > 2 else 0.0
    mediana = float(s.median())
    escala = float(s.abs().max()) / (abs(mediana) if mediana else 1.0)
    hay_negativos = bool((s < 0).any())

    if hay_negativos:
        if asimetria > 2 or escala > 10:
            return "yeojohnson"
        return "robust"
    if asimetria > 2 or escala > 10:
        return "log1p"
    if float(s.min()) >= 0 and float(s.max()) <= 1:
        return "minmax"
    if escala > 3:
        return "robust"
    return "zscore"


def split_scale(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "zscore",
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, Normalizer]:
    """Ajusta el escalador solo con ``train`` y aplica a ambos.

    Evita la fuga de informacion (data leakage) al preparar modelos.
    """
    esc = Normalizer(method, columns, **kwargs).fit(train)
    return esc.transform(train), esc.transform(test), esc
