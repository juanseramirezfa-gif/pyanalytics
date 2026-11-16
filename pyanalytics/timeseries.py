"""Analisis de series temporales y pronostico.

Cubre remuestreo, medias moviles, diferenciacion, pruebas de estacionariedad,
descomposicion, autocorrelacion y varios modelos de pronostico
(ingenuo, suavizado exponencial Holt-Winters y SARIMA).

Ejemplos
--------
>>> from pyanalytics.timeseries import resample_series, forecast, metrics
>>> mensual = resample_series(df, "fecha", "pm25_ug_m3", freq="MS", agg="mean")
>>> pred = forecast(mensual["pm25_ug_m3"], horizon=12, method="holt_winters")
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "to_datetime_index",
    "resample_series",
    "rolling_stats",
    "difference",
    "stationarity_tests",
    "acf_pacf",
    "decompose",
    "seasonal_strength",
    "train_test_split_ts",
    "forecast",
    "metrics",
    "backtest",
    "describe_series",
]

ForecastMethod = Literal["naive", "seasonal_naive", "mean", "drift", "ses", "holt", "holt_winters", "sarima", "auto"]


def to_datetime_index(
    df: pd.DataFrame,
    date_column: str,
    value_columns: str | Sequence[str] | None = None,
    *,
    freq: str | None = None,
    agg: str = "sum",
    dropna: bool = True,
) -> pd.DataFrame:
    """Convierte una columna de fecha en indice temporal y remuestrea si se pide.

    ``freq`` usa alias de pandas: ``"D"`` diario, ``"W"`` semanal, ``"MS"``
    inicio de mes, ``"QS"`` trimestral, ``"YS"`` anual, ``"h"`` horario.
    """
    out = df.copy()
    out[date_column] = pd.to_datetime(out[date_column], errors="coerce", format="mixed")
    out = out.dropna(subset=[date_column]).sort_values(date_column)
    if value_columns is None:
        valores = [c for c in out.columns if c != date_column and pd.api.types.is_numeric_dtype(out[c])]
    elif isinstance(value_columns, str):
        valores = [value_columns]
    else:
        valores = list(value_columns)
    out = out.set_index(date_column)[valores]
    if freq:
        out = out.resample(freq).agg(agg)
    if dropna:
        out = out.dropna(how="all")
    return out


def resample_series(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    *,
    freq: str = "MS",
    agg: str | Sequence[str] = "sum",
) -> pd.DataFrame:
    """Agrega una serie a la frecuencia indicada."""
    return to_datetime_index(df, date_column, value_column, freq=freq, agg=agg)


def rolling_stats(
    serie: pd.Series,
    windows: Sequence[int] = (7, 30),
    aggs: Sequence[str] = ("mean", "std", "min", "max"),
) -> pd.DataFrame:
    """Medias/desviaciones moviles de una serie."""
    out = pd.DataFrame({"valor": serie})
    for w in windows:
        roll = serie.rolling(window=w, min_periods=max(1, w // 2))
        for agg in aggs:
            out[f"roll{w}_{agg}"] = getattr(roll, agg)()
    out["expanding_mean"] = serie.expanding().mean()
    return out


def difference(serie: pd.Series, order: int = 1, seasonal: int | None = None) -> pd.Series:
    """Diferenciacion simple y/o estacional para estabilizar la serie."""
    out = serie.astype(float)
    if seasonal:
        out = out.diff(seasonal)
    for _ in range(order):
        out = out.diff()
    return out


def stationarity_tests(serie: pd.Series, regression: str = "c") -> dict[str, Any]:
    """Pruebas de raiz unitaria: Dickey-Fuller aumentada (ADF) y KPSS."""
    from statsmodels.tsa.stattools import adfuller, kpss

    s = serie.dropna().astype(float)
    resultado: dict[str, Any] = {"n_observaciones": int(len(s))}
    if len(s) < 8:
        resultado["error"] = "Se necesitan al menos 8 observaciones."
        return resultado

    try:
        salida_adf = _llamar_con_opcion(
            adfuller, s, opcion=("result_object", False), autolag="AIC", regression=regression
        )
        adf_stat, adf_p, lags, nobs, criticos, _ = salida_adf
        resultado.update(
            {
                "adf_estadistico": round(float(adf_stat), 4),
                "adf_pvalor": round(float(adf_p), 4),
                "adf_lags": int(lags),
                "adf_criticos": {k: round(float(v), 4) for k, v in criticos.items()},
                "adf_estacionaria": bool(adf_p < 0.05),
            }
        )
    except Exception as exc:  # pragma: no cover
        resultado["adf_error"] = str(exc)

    try:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            salida_kpss = _llamar_con_opcion(
                kpss,
                s,
                opcion=("result_object", False),
                regression="c" if regression == "c" else "ct",
                nlags="auto",
            )
        kpss_stat, kpss_p, kpss_lags, criticos_kpss = salida_kpss
        resultado.update(
            {
                "kpss_estadistico": round(float(kpss_stat), 4),
                "kpss_pvalor": round(float(kpss_p), 4),
                "kpss_lags": int(kpss_lags),
                "kpss_criticos": {k: round(float(v), 4) for k, v in criticos_kpss.items()},
                "kpss_estacionaria": bool(kpss_p > 0.05),
            }
        )
    except Exception as exc:  # pragma: no cover
        resultado["kpss_error"] = str(exc)

    if "adf_estacionaria" in resultado and "kpss_estacionaria" in resultado:
        resultado["veredicto"] = (
            "estacionaria" if resultado["adf_estacionaria"] and resultado["kpss_estacionaria"]
            else "no estacionaria" if not resultado["adf_estacionaria"] and not resultado["kpss_estacionaria"]
            else "resultado mixto (revisar)"
        )
    return resultado


def acf_pacf(serie: pd.Series, lags: int | None = None) -> dict[str, pd.DataFrame]:
    """Autocorrelacion (ACF) y autocorrelacion parcial (PACF)."""
    from statsmodels.tsa.stattools import acf, pacf

    s = serie.dropna().astype(float)
    if lags is None:
        lags = min(40, max(1, len(s) // 3))
    valores_acf = acf(s, nlags=lags, fft=True)
    valores_pacf = pacf(s, nlags=lags, method="ywm")
    ic = 1.96 / np.sqrt(len(s))
    return {
        "acf": pd.DataFrame({"lag": range(len(valores_acf)), "acf": valores_acf, "ic_superior": ic, "ic_inferior": -ic}),
        "pacf": pd.DataFrame({"lag": range(len(valores_pacf)), "pacf": valores_pacf, "ic_superior": ic, "ic_inferior": -ic}),
    }


def acf_pacf_plot(serie: pd.Series, lags: int | None = None, figsize: Sequence[float] = (11, 4)):
    """Grafico conjunto de ACF y PACF."""
    import matplotlib.pyplot as plt

    from .plots import _ensure_style

    _ensure_style()
    datos = acf_pacf(serie, lags)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].stem(datos["acf"]["lag"], datos["acf"]["acf"], basefmt=" ")
    axes[0].axhline(datos["acf"]["ic_superior"].iloc[0], ls="--", color="gray")
    axes[0].axhline(datos["acf"]["ic_inferior"].iloc[0], ls="--", color="gray")
    axes[0].set_title("ACF")
    axes[1].stem(datos["pacf"]["lag"], datos["pacf"]["pacf"], basefmt=" ")
    axes[1].axhline(datos["pacf"]["ic_superior"].iloc[0], ls="--", color="gray")
    axes[1].axhline(datos["pacf"]["ic_inferior"].iloc[0], ls="--", color="gray")
    axes[1].set_title("PACF")
    fig.suptitle("Autocorrelacion")
    return fig


def decompose(serie: pd.Series, period: int | None = None, model: Literal["additive", "multiplicative"] = "additive", figsize: Sequence[float] = (11, 7)):
    """Descompone la serie en tendencia, estacionalidad y residuo."""
    import matplotlib.pyplot as plt
    from statsmodels.tsa.seasonal import seasonal_decompose

    from .plots import _ensure_style

    _ensure_style()
    s = serie.dropna().astype(float)
    if period is None:
        period = _infer_period(s.index)
    if period < 2 or len(s) < 2 * period:
        raise ValueError(f"Periodo invalido ({period}) para {len(s)} observaciones.")
    resultado = seasonal_decompose(s, model=model, period=period, extrapolate_trend="period")
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    axes[0].plot(s, color="steelblue")
    axes[0].set_ylabel("Observado")
    axes[1].plot(resultado.trend, color="darkorange")
    axes[1].set_ylabel("Tendencia")
    axes[2].plot(resultado.seasonal, color="seagreen")
    axes[2].set_ylabel("Estacional")
    axes[3].plot(resultado.resid, color="crimson")
    axes[3].set_ylabel("Residuo")
    fig.suptitle(f"Descomposicion ({model}, periodo={period})")
    return fig


def seasonal_strength(serie: pd.Series, period: int | None = None, model: Literal["additive", "multiplicative"] = "additive") -> dict[str, float]:
    """Fuerza de la estacionalidad y de la tendencia (0-1, como en STL)."""
    from statsmodels.tsa.seasonal import seasonal_decompose

    s = serie.dropna().astype(float)
    if period is None:
        period = _infer_period(s.index)
    if period < 2 or len(s) < 2 * period:
        return {"estacionalidad": float("nan"), "tendencia": float("nan")}
    res = seasonal_decompose(s, model=model, period=period, extrapolate_trend="period")
    resid = res.resid.dropna()
    if resid.empty:
        return {"estacionalidad": float("nan"), "tendencia": float("nan")}
    var_resid = float(np.var(resid))
    var_seasonal = float(np.var(res.seasonal.dropna() + resid))
    var_trend = float(np.var(res.trend.dropna() + resid))
    fuerza_est = max(0.0, 1 - var_resid / var_seasonal) if var_seasonal else 0.0
    fuerza_tend = max(0.0, 1 - var_resid / var_trend) if var_trend else 0.0
    return {"estacionalidad": round(fuerza_est, 4), "tendencia": round(fuerza_tend, 4)}


def train_test_split_ts(serie: pd.Series, test_size: int | float = 0.2) -> tuple[pd.Series, pd.Series]:
    """Divide una serie respetando el orden temporal (sin mezclar)."""
    s = serie.dropna()
    n_test = int(round(test_size * len(s))) if isinstance(test_size, float) else int(test_size)
    n_test = max(1, min(n_test, len(s) - 1))
    return s.iloc[:-n_test], s.iloc[-n_test:]


def forecast(
    serie: pd.Series,
    horizon: int = 12,
    *,
    method: ForecastMethod = "auto",
    seasonality: int | None = None,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] | None = None,
    return_confint: bool = True,
) -> pd.DataFrame:
    """Pronostica los siguientes ``horizon`` periodos.

    Metodos: ``naive``, ``seasonal_naive``, ``mean``, ``drift``, ``ses``,
    ``holt``, ``holt_winters``, ``sarima`` y ``auto`` (elige segun la serie).

    Devuelve un DataFrame con ``pronostico``, ``limite_inferior`` y
    ``limite_superior`` indexado por fecha futura.
    """
    s = serie.dropna().astype(float).sort_index()
    if len(s) < 4:
        raise ValueError("Se necesitan al menos 4 observaciones para pronosticar.")
    # statsmodels exige un indice con frecuencia conocida; si la serie tiene
    # huecos (p. ej. al eliminar nulos) usamos un indice entero para el ajuste y
    # reconstruimos las fechas futuras con _future_index.
    s_modelo = s.reset_index(drop=True)
    s_modelo.index = pd.RangeIndex(len(s_modelo))
    if seasonality is None:
        seasonality = _infer_period(s.index)
    seasonality = max(2, int(seasonality))

    if method == "auto":
        if len(s) >= 2 * seasonality and seasonality >= 2:
            method = "holt_winters"
        else:
            method = "holt"

    if method == "naive":
        valores = np.repeat(s.iloc[-1], horizon)
        return _frame_forecast(s.index, horizon, valores, None)

    if method == "seasonal_naive":
        patron = s.iloc[-seasonality:].to_numpy()
        valores = np.resize(patron, horizon)
        return _frame_forecast(s.index, horizon, valores, None)

    if method == "mean":
        valores = np.repeat(s.mean(), horizon)
        return _frame_forecast(s.index, horizon, valores, None)

    if method == "drift":
        pendiente = (s.iloc[-1] - s.iloc[0]) / (len(s) - 1)
        valores = s.iloc[-1] + pendiente * np.arange(1, horizon + 1)
        return _frame_forecast(s.index, horizon, valores, None)

    if method == "ses":
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing

        modelo = SimpleExpSmoothing(s_modelo, initialization_method="estimated").fit()
        pred = modelo.forecast(horizon)
        return _frame_forecast(s.index, horizon, pred.to_numpy(), None)

    if method == "holt":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        modelo = ExponentialSmoothing(s_modelo, trend="add", initialization_method="estimated").fit()
        pred = modelo.forecast(horizon)
        return _frame_forecast(s.index, horizon, pred.to_numpy(), None)

    if method == "holt_winters":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        estacional = "add"
        modelo = ExponentialSmoothing(
            s_modelo,
            trend="add",
            seasonal=estacional,
            seasonal_periods=seasonality,
            initialization_method="estimated",
        ).fit()
        pred = modelo.forecast(horizon)
        ic = _residual_confint(modelo, pred) if return_confint else None
        return _frame_forecast(s.index, horizon, pred.to_numpy(), ic)

    if method == "sarima":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        sorder = seasonal_order if seasonal_order is not None else (0, 1, 1, seasonality)
        modelo = SARIMAX(
            s_modelo,
            order=order,
            seasonal_order=sorder,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        pred = modelo.get_forecast(steps=horizon)
        media = pred.predicted_mean
        ic = pred.conf_int(alpha=0.05)
        return _frame_forecast(s.index, horizon, media.to_numpy(), ic.to_numpy() if return_confint else None)

    raise ValueError(f"Metodo de pronostico no soportado: {method!r}")


def metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    """Metricas de error de pronostico: MAE, RMSE, MAPE, sMAPE y R2."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mascara = ~(np.isnan(y) | np.isnan(p))
    y, p = y[mascara], p[mascara]
    if len(y) == 0:
        return {}
    error = y - p
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    no_cero = y != 0
    mape = float(np.mean(np.abs(error[no_cero] / y[no_cero])) * 100) if no_cero.any() else float("nan")
    denom = (np.abs(y) + np.abs(p)) / 2
    validos = denom != 0
    smape = float(np.mean(np.abs(error[validos] / denom[validos])) * 100) if validos.any() else float("nan")
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4),
        "smape": round(smape, 4),
        "r2": round(r2, 4),
        "n": int(len(y)),
    }


def backtest(
    serie: pd.Series,
    *,
    horizon: int = 6,
    n_splits: int = 3,
    method: ForecastMethod = "holt",
    seasonality: int | None = None,
) -> pd.DataFrame:
    """Validacion retrospectiva: entrena en el pasado y evalua contra lo real."""
    s = serie.dropna().astype(float).sort_index()
    filas: list[dict[str, Any]] = []
    total = len(s)
    for i in range(n_splits, 0, -1):
        corte = total - i * horizon
        if corte < max(6, horizon):
            continue
        train, test = s.iloc[:corte], s.iloc[corte : corte + horizon]
        try:
            pred = forecast(train, horizon=len(test), method=method, seasonality=seasonality, return_confint=False)
        except Exception as exc:
            filas.append({"corte": str(train.index[-1]), "error": str(exc)})
            continue
        valores = pred["pronostico"].to_numpy()
        fila: dict[str, Any] = {"corte": str(train.index[-1]), "metodo": method}
        fila.update(metrics(test.to_numpy(), valores))
        filas.append(fila)
    out = pd.DataFrame(filas)
    if not out.empty and "mae" in out.columns:
        promedios = {c: round(float(out[c].mean()), 4) for c in ("mae", "rmse", "mape", "smape") if c in out.columns}
        out.attrs["promedio"] = promedios
    return out


def describe_series(serie: pd.Series, period: int | None = None) -> dict[str, Any]:
    """Resumen completo de una serie temporal: frecuencia, estacionariedad, etc."""
    s = serie.dropna().astype(float).sort_index()
    if period is None:
        period = _infer_period(s.index)
    info: dict[str, Any] = {
        "n_observaciones": int(len(s)),
        "inicio": str(s.index[0]) if len(s) else None,
        "fin": str(s.index[-1]) if len(s) else None,
        "frecuencia_inferida": _freq_alias(s.index),
        "periodo_estacional": int(period),
        "media": round(float(s.mean()), 4) if len(s) else None,
        "desviacion": round(float(s.std()), 4) if len(s) else None,
        "min": round(float(s.min()), 4) if len(s) else None,
        "max": round(float(s.max()), 4) if len(s) else None,
        "nulos": int(serie.isna().sum()),
    }
    info.update({k: v for k, v in stationarity_tests(s).items() if k in ("adf_pvalor", "adf_estacionaria", "veredicto")})
    if len(s) >= 2 * period:
        info.update(seasonal_strength(s, period))
    return info


# --------------------------------------------------------------------------- #
# utilidades internas
# --------------------------------------------------------------------------- #
def _frame_forecast(index: pd.Index, horizon: int, valores: np.ndarray, ic: np.ndarray | None) -> pd.DataFrame:
    futuro = _future_index(index, horizon)
    out = pd.DataFrame({"pronostico": np.asarray(valores, dtype=float)}, index=futuro)
    if ic is not None and np.asarray(ic).size:
        ic = np.asarray(ic, dtype=float)
        out["limite_inferior"] = ic[:, 0]
        out["limite_superior"] = ic[:, 1]
    out.index.name = index.name or "fecha"
    return out


def _residual_confint(resultado: Any, pred: Any, alpha: float = 0.05) -> np.ndarray:
    """Intervalo de confianza aproximado a partir de los residuos del modelo.

    Se usa cuando el modelo no expone ``get_prediction`` (p. ej. Holt-Winters
    en statsmodels >= 0.15).
    """
    from scipy import stats as _stats

    residuos = np.asarray(getattr(resultado, "resid", []), dtype=float)
    residuos = residuos[~np.isnan(residuos)]
    sigma = float(residuos.std(ddof=1)) if len(residuos) > 1 else 0.0
    z = float(_stats.norm.ppf(1 - alpha / 2))
    valores = np.asarray(pred, dtype=float)
    return np.column_stack([valores - z * sigma, valores + z * sigma])


def _llamar_con_opcion(func: Any, *args: Any, opcion: tuple[str, Any], **kwargs: Any) -> Any:
    """Llama a ``func`` con una opcion nueva, con respaldo para versiones antiguas.

    Permite usar ``result_object=False`` en statsmodels >= 0.15 sin romper en
    versiones anteriores que no aceptan ese argumento.
    """
    nombre, valor = opcion
    try:
        return func(*args, **{nombre: valor}, **kwargs)
    except TypeError:
        return func(*args, **kwargs)


def _infer_freq(index: pd.Index) -> str | None:
    """Frecuencia inferida como texto, compatible con pandas 2 y 3."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return None
    try:
        freq = pd.infer_freq(index)
    except Exception:
        return None
    if freq is None:
        return None
    return freq if isinstance(freq, str) else getattr(freq, "freqstr", None)


def _future_index(index: pd.Index, horizon: int) -> pd.DatetimeIndex:
    freq = _infer_freq(index)
    if isinstance(index, pd.DatetimeIndex):
        if freq:
            return pd.date_range(start=index[-1], periods=horizon + 1, freq=freq)[1:]
        paso = (index[-1] - index[-2]) if len(index) >= 2 else pd.Timedelta(days=1)
        return pd.DatetimeIndex([index[-1] + paso * (i + 1) for i in range(horizon)])
    return pd.Index(range(len(index), len(index) + horizon), name=index.name)


def _infer_period(index: pd.Index) -> int:
    if isinstance(index, pd.DatetimeIndex) and len(index) >= 3:
        freq = _infer_freq(index)
        if freq:
            base = freq[0].upper()
            if base == "D":
                return 7
            if base == "W":
                return 52
            if base in ("M", "B"):
                return 12
            if base == "Q":
                return 4
            if base == "A" or base == "Y":
                return 1
            if base in ("H", "h"):
                return 24
    return 2


def _freq_alias(index: pd.Index) -> str | None:
    if isinstance(index, pd.DatetimeIndex) and len(index) >= 3:
        return pd.infer_freq(index)
    return None
