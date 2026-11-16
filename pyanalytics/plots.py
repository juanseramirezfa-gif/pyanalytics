"""Graficos estaticos (matplotlib/seaborn) e interactivos (plotly).

Cada funcion devuelve un objeto ``Figure`` para que puedas mostrarlo,
guardarlo o incrustarlo en un informe.

Ejemplos
--------
>>> from pyanalytics.plots import histogram, heatmap_corr, save_figure
>>> fig = histogram(df, "precio")
>>> save_figure(fig, "outputs/precio.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import matplotlib

matplotlib.use("Agg")  # backend sin pantalla (servidores, scripts)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .profile import categorical_columns, correlation_matrix, datetime_columns, numeric_columns

__all__ = [
    "PALETTE",
    "set_style",
    "histogram",
    "kde_plot",
    "ecdf_plot",
    "boxplot",
    "violinplot",
    "scatter",
    "line",
    "barplot",
    "countplot",
    "pie",
    "heatmap_corr",
    "pairplot",
    "missing_matrix",
    "qq_plot",
    "area",
    "regression_plot",
    "auto_plots",
    "save_figure",
    "figure_to_base64",
    "interactive_scatter",
    "interactive_line",
    "interactive_histogram",
    "interactive_box",
    "interactive_corr",
]

PALETTE = "viridis"
_STYLE_DONE = False


def set_style(style: str = "whitegrid", palette: str = PALETTE, context: str = "notebook") -> None:
    """Configura el estilo global de los graficos."""
    global _STYLE_DONE
    sns.set_theme(style=style, palette=palette, context=context)
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.max_open_warning"] = 0  # informes con muchos graficos
    _STYLE_DONE = True


def _ensure_style() -> None:
    if not _STYLE_DONE:
        set_style()


def _figsize(figsize: Sequence[float] | None) -> tuple[float, float]:
    return tuple(figsize) if figsize else (10, 5)  # type: ignore[return-value]


def histogram(
    df: pd.DataFrame,
    column: str,
    *,
    bins: int | str = "auto",
    kde: bool = True,
    by: str | None = None,
    figsize: Sequence[float] | None = None,
    title: str | None = None,
):
    """Histograma de una columna numerica (con curva KDE opcional)."""
    _ensure_style()
    fig, ax = plt.subplots(figsize=_figsize(figsize))
    sns.histplot(data=df, x=column, bins=bins, kde=kde, hue=by, ax=ax, edgecolor="white")
    ax.set_title(title or f"Distribucion de {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frecuencia")
    return fig


def kde_plot(
    df: pd.DataFrame,
    column: str,
    *,
    by: str | None = None,
    fill: bool = True,
    figsize: Sequence[float] | None = None,
):
    """Curva de densidad (KDE)."""
    _ensure_style()
    fig, ax = plt.subplots(figsize=_figsize(figsize))
    sns.kdeplot(data=df, x=column, hue=by, fill=fill, ax=ax, common_norm=False)
    ax.set_title(f"Densidad de {column}")
    return fig


def ecdf_plot(df: pd.DataFrame, column: str, *, by: str | None = None, figsize: Sequence[float] | None = None):
    """Funcion de distribucion acumulada empirica."""
    _ensure_style()
    fig, ax = plt.subplots(figsize=_figsize(figsize))
    sns.ecdfplot(data=df, x=column, hue=by, ax=ax)
    ax.set_title(f"Distribucion acumulada de {column}")
    return fig


def boxplot(
    df: pd.DataFrame,
    column: str,
    *,
    by: str | None = None,
    horizontal: bool = False,
    figsize: Sequence[float] | None = None,
):
    """Diagrama de caja (detecta atipicos visualmente)."""
    _ensure_style()
    fig, ax = plt.subplots(figsize=_figsize(figsize))
    if horizontal:
        sns.boxplot(data=df, x=column, y=by, ax=ax)
    else:
        sns.boxplot(data=df, y=column, x=by, ax=ax)
    ax.set_title(f"Caja de {column}" + (f" por {by}" if by else ""))
    return fig


def violinplot(
    df: pd.DataFrame,
    column: str,
    *,
    by: str | None = None,
    figsize: Sequence[float] | None = None,
):
    """Diagrama de violin (distribucion + densidad)."""
    _ensure_style()
    fig, ax = plt.subplots(figsize=_figsize(figsize))
    sns.violinplot(data=df, y=column, x=by, ax=ax, inner="quart")
    ax.set_title(f"Violin de {column}" + (f" por {by}" if by else ""))
    return fig


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    hue: str | None = None,
    size: str | None = None,
    trend: bool = False,
    figsize: Sequence[float] | None = None,
):
    """Diagrama de dispersion entre dos variables."""
    _ensure_style()
    fig, ax = plt.subplots(figsize=_figsize(figsize))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, size=size, ax=ax, alpha=0.75)
    if trend:
        sns.regplot(data=df, x=x, y=y, ax=ax, scatter=False, color="crimson", line_kws={"lw": 2})
    ax.set_title(f"{y} vs {x}")
    return fig


def line(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    hue: str | None = None,
    figsize: Sequence[float] | None = None,
    markers: bool = False,
):
    """Grafico de lineas (ideal para series temporales)."""
    _ensure_style()
    fig, ax = plt.subplots(figsize=_figsize(figsize))
    sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax, marker="o" if markers else None)
    ax.set_title(f"{y} a lo largo de {x}")
    return fig


def barplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    hue: str | None = None,
    estimator: str = "mean",
    horizontal: bool = False,
    top: int | None = None,
    figsize: Sequence[float] | None = None,
):
    """Barras de un agregado de ``y`` por categoria ``x``."""
    _ensure_style()
    datos = df
    if top is not None:
        orden = df.groupby(x, dropna=False)[y].mean().sort_values(ascending=False).head(top).index
        datos = df[df[x].isin(orden)]
    fig, ax = plt.subplots(figsize=_figsize(figsize))
    if horizontal:
        sns.barplot(data=datos, y=x, x=y, hue=hue, estimator=estimator, ax=ax)
    else:
        sns.barplot(data=datos, x=x, y=y, hue=hue, estimator=estimator, ax=ax)
    ax.set_title(f"{estimator} de {y} por {x}")
    if not horizontal and datos[x].nunique() > 8:
        ax.tick_params(axis="x", rotation=45)
    return fig


def countplot(
    df: pd.DataFrame,
    column: str,
    *,
    hue: str | None = None,
    top: int | None = None,
    horizontal: bool = False,
    figsize: Sequence[float] | None = None,
):
    """Conteo de frecuencias de una variable categorica."""
    _ensure_style()
    datos = df
    if top is not None:
        top_vals = df[column].value_counts().head(top).index
        datos = df[df[column].isin(top_vals)]
    fig, ax = plt.subplots(figsize=_figsize(figsize))
    if horizontal:
        sns.countplot(data=datos, y=column, hue=hue, ax=ax, order=datos[column].value_counts().index)
    else:
        sns.countplot(data=datos, x=column, hue=hue, ax=ax, order=datos[column].value_counts().index)
        if datos[column].nunique() > 8:
            ax.tick_params(axis="x", rotation=45)
    ax.set_title(f"Frecuencia de {column}")
    return fig


def pie(
    df: pd.DataFrame,
    column: str,
    *,
    top: int = 10,
    figsize: Sequence[float] | None = None,
):
    """Grafico circular de participacion por categoria."""
    _ensure_style()
    vc = df[column].value_counts().head(top)
    if len(df[column].value_counts()) > top:
        vc["__otras__"] = df[column].value_counts().iloc[top:].sum()
    fig, ax = plt.subplots(figsize=_figsize(figsize or (7, 7)))
    ax.pie(vc.values, labels=vc.index.astype(str), autopct="%1.1f%%", startangle=90, counterclock=False)
    ax.set_title(f"Composicion de {column}")
    ax.axis("equal")
    return fig


def heatmap_corr(
    df: pd.DataFrame,
    *,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    annot: bool | None = None,
    figsize: Sequence[float] | None = None,
    columns: list[str] | None = None,
):
    """Mapa de calor de correlaciones."""
    _ensure_style()
    corr = correlation_matrix(df if columns is None else df[columns], method=method)
    if corr.empty:
        raise ValueError("Se necesitan al menos dos columnas numericas.")
    if annot is None:
        annot = corr.shape[0] <= 12
    fig, ax = plt.subplots(figsize=_figsize(figsize or (max(6, corr.shape[0] * 0.8), max(5, corr.shape[0] * 0.7))))
    sns.heatmap(corr, annot=annot, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title(f"Correlaciones ({method})")
    return fig


def pairplot(
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    hue: str | None = None,
    max_columns: int = 6,
):
    """Matriz de dispersion entre pares de variables."""
    _ensure_style()
    cols = columns or numeric_columns(df)
    cols = cols[:max_columns]
    if len(cols) < 2:
        raise ValueError("Se necesitan al menos dos columnas numericas.")
    grid = sns.pairplot(df[cols + ([hue] if hue and hue not in cols else [])].dropna(), hue=hue, corner=True)
    grid.figure.suptitle("Matriz de dispersion", y=1.02)
    return grid.figure


def missing_matrix(df: pd.DataFrame, *, figsize: Sequence[float] | None = None):
    """Mapa de valores nulos (patron de ausencia)."""
    _ensure_style()
    fig, ax = plt.subplots(figsize=_figsize(figsize or (10, max(3, len(df.columns) * 0.3))))
    sns.heatmap(df.isna().T, cbar=False, cmap=["#d9d9d9", "#d62728"], ax=ax)
    ax.set_title("Mapa de valores nulos")
    ax.set_xlabel("Filas")
    ax.set_ylabel("Columnas")
    return fig


def qq_plot(df: pd.DataFrame, column: str, *, figsize: Sequence[float] | None = None):
    """Grafico cuantil-cuantil contra la normal."""
    from scipy import stats

    _ensure_style()
    datos = pd.to_numeric(df[column], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=_figsize(figsize))
    stats.probplot(datos, dist="norm", plot=ax)
    ax.set_title(f"QQ-plot de {column}")
    return fig


def area(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    hue: str | None = None,
    figsize: Sequence[float] | None = None,
):
    """Grafico de area apilada."""
    _ensure_style()
    fig, ax = plt.subplots(figsize=_figsize(figsize))
    if hue:
        pivote = df.pivot_table(index=x, columns=hue, values=y, aggfunc="sum").fillna(0)
        pivote.plot.area(ax=ax, stacked=True, alpha=0.85)
    else:
        ax.fill_between(df[x], df[y], alpha=0.4)
        ax.plot(df[x], df[y])
    ax.set_title(f"Area de {y} sobre {x}")
    return fig


def regression_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    order: int = 1,
    figsize: Sequence[float] | None = None,
):
    """Dispersion con recta/curva de ajuste y banda de confianza."""
    _ensure_style()
    fig, ax = plt.subplots(figsize=_figsize(figsize))
    sns.regplot(data=df, x=x, y=y, ax=ax, order=order, scatter_kws={"alpha": 0.6})
    r = df[[x, y]].corr().iloc[0, 1]
    ax.set_title(f"{y} vs {x}  (r = {r:.3f})")
    return fig


def auto_plots(
    df: pd.DataFrame,
    *,
    target: str | None = None,
    max_numeric: int = 6,
    max_categorical: int = 6,
) -> dict[str, Any]:
    """Genera un conjunto razonable de graficos segun los tipos de columna.

    Devuelve un diccionario ``{nombre: figura}``.
    """
    _ensure_style()
    figuras: dict[str, Any] = {}
    num = numeric_columns(df)[:max_numeric]
    cat = categorical_columns(df, max_unique=25)[:max_categorical]
    fechas = datetime_columns(df)

    for col in num:
        try:
            figuras[f"hist_{col}"] = histogram(df, col)
        except Exception:
            pass
    for col in cat:
        try:
            figuras[f"bar_{col}"] = countplot(df, col, top=15)
        except Exception:
            pass
    if len(num) >= 2:
        try:
            figuras["correlaciones"] = heatmap_corr(df, columns=num)
        except Exception:
            pass
    if len(num) >= 2:
        try:
            figuras["pares"] = pairplot(df, columns=num[:5])
        except Exception:
            pass
    if target and target in num:
        for col in [c for c in num if c != target][:3]:
            try:
                figuras[f"disp_{col}_vs_{target}"] = scatter(df, col, target, trend=True)
            except Exception:
                pass
    if fechas and num:
        try:
            figuras[f"serie_{num[0]}"] = line(df.sort_values(fechas[0]), fechas[0], num[0])
        except Exception:
            pass
    if df.isna().any().any():
        figuras["nulos"] = missing_matrix(df)
    return figuras


def save_figure(fig: Any, path: str | Path, *, dpi: int = 150, transparent: bool = False) -> Path:
    """Guarda una figura en disco y devuelve la ruta."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=transparent)
    return path


def figure_to_base64(fig: Any, *, format: str = "png", dpi: int = 110) -> str:
    """Convierte una figura a una cadena base64 (para incrustar en HTML)."""
    import base64
    import io

    buffer = io.BytesIO()
    fig.savefig(buffer, format=format, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- #
# versiones interactivas (plotly) para Streamlit
# --------------------------------------------------------------------------- #
def interactive_scatter(df: pd.DataFrame, x: str, y: str, *, color: str | None = None, trend: bool = False):
    """Dispersion interactiva con plotly."""
    import plotly.express as px

    fig = px.scatter(df, x=x, y=y, color=color, trendline="ols" if trend else None, opacity=0.75)
    fig.update_layout(title=f"{y} vs {x}", template="plotly_white")
    return fig


def interactive_line(df: pd.DataFrame, x: str, y: str, *, color: str | None = None):
    """Lineas interactivas con plotly."""
    import plotly.express as px

    fig = px.line(df, x=x, y=y, color=color, markers=True)
    fig.update_layout(title=f"{y} a lo largo de {x}", template="plotly_white")
    return fig


def interactive_histogram(df: pd.DataFrame, column: str, *, color: str | None = None, nbins: int = 40):
    """Histograma interactivo con plotly."""
    import plotly.express as px

    fig = px.histogram(df, x=column, color=color, nbins=nbins, marginal="box")
    fig.update_layout(title=f"Distribucion de {column}", template="plotly_white")
    return fig


def interactive_box(df: pd.DataFrame, column: str, *, by: str | None = None):
    """Caja interactiva con plotly."""
    import plotly.express as px

    fig = px.box(df, y=column, x=by, color=by, points="outliers")
    fig.update_layout(title=f"Caja de {column}", template="plotly_white", showlegend=False)
    return fig


def interactive_corr(df: pd.DataFrame, *, method: str = "pearson"):
    """Mapa de calor de correlaciones interactivo."""
    import plotly.express as px

    corr = correlation_matrix(df, method=method)  # type: ignore[arg-type]
    if corr.empty:
        raise ValueError("Se necesitan al menos dos columnas numericas.")
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(title=f"Correlaciones ({method})", template="plotly_white")
    return fig
