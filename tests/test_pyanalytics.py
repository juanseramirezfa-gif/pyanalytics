"""Pruebas del paquete pyanalytics.

Ejecutar:
    .venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

import pyanalytics as pa


@pytest.fixture()
def df_sucio() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame(
        {
            "ID Cliente": range(1, n + 1),
            "Fecha Venta": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
            "Ciudad ": rng.choice(["Bogota", "Medellin", " Cali "], n),
            "Categoria": rng.choice(["a", "b", "c"], n),
            "Precio Unitario": np.round(rng.gamma(2, 100, n), 2),
            "Cantidad": rng.integers(1, 10, n),
            "Descuento": rng.choice([0, 5, 10, None], n),
            "Satisfaccion": np.round(rng.normal(4, 0.6, n), 2),
        }
    )
    df.loc[rng.choice(n, 10, replace=False), "Satisfaccion"] = np.nan
    df.loc[rng.choice(n, 5, replace=False), "Precio Unitario"] = np.nan
    df = pd.concat([df, df.iloc[:8]], ignore_index=True)
    return df


# --------------------------------------------------------------------------- #
def test_io_ida_y_vuelta(tmp_path, df_sucio: pd.DataFrame) -> None:
    limpio = df_sucio.dropna()
    for extension in (".csv", ".xlsx", ".parquet", ".json"):
        ruta = tmp_path / f"datos{extension}"
        pa.save_data(limpio, ruta)
        assert ruta.exists()
        leido = pa.load_data(ruta)
        assert leido.shape[0] == limpio.shape[0]
        assert leido.shape[1] == limpio.shape[1]


def test_listar_hojas_excel(tmp_path, df_sucio: pd.DataFrame) -> None:
    ruta = tmp_path / "libro.xlsx"
    with pd.ExcelWriter(ruta) as writer:
        df_sucio.to_excel(writer, index=False, sheet_name="Hoja1")
    assert pa.list_excel_sheets(ruta) == ["Hoja1"]


def test_formato_no_soportado(tmp_path) -> None:
    ruta = tmp_path / "raro.xyz"
    ruta.write_text("hola")
    with pytest.raises(ValueError):
        pa.load_data(ruta)


# --------------------------------------------------------------------------- #
def test_perfil(df_sucio: pd.DataFrame) -> None:
    perfil = pa.full_profile(df_sucio)
    ov = perfil["overview"]
    assert ov["filas"] == len(df_sucio)
    assert ov["columnas"] == df_sucio.shape[1]
    assert ov["filas_duplicadas"] == 8
    assert 0 <= perfil["calidad"]["puntaje"] <= 100
    assert set(perfil) >= {"columnas", "nulos", "correlaciones", "atipicos", "calidad"}


def test_columnas_por_tipo(df_sucio: pd.DataFrame) -> None:
    assert "Precio Unitario" in pa.numeric_columns(df_sucio)
    assert "Ciudad " in pa.categorical_columns(df_sucio)
    tipos = pa.column_types(df_sucio)
    assert "Cantidad" in tipos["numericas"]


def test_deteccion_atipicos() -> None:
    serie = pd.DataFrame({"x": [1, 1, 1, 2, 2, 2, 3, 3, 100]})
    assert pa.detect_outliers(serie, "x").sum() == 1


def test_outliers_report_sin_columnas_numericas() -> None:
    """Regresion: una tabla solo de texto no debe romper el perfilado."""
    df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "p"]})
    informe = pa.outliers_report(df)
    assert informe.empty
    assert list(informe.columns) == [
        "columna",
        "metodo",
        "n_atipicos",
        "pct_atipicos",
        "limite_inferior",
        "limite_superior",
    ]
    assert pa.full_profile(df)["atipicos"].empty


def test_outliers_report_dataframe_vacio() -> None:
    informe = pa.outliers_report(pd.DataFrame())
    assert informe.empty
    assert "n_atipicos" in informe.columns


def test_correlaciones() -> None:
    df = pd.DataFrame({"a": range(50), "b": [i * 2 for i in range(50)], "c": np.random.default_rng(1).normal(size=50)})
    altas = pa.high_correlations(df, threshold=0.9)
    assert len(altas) == 1
    assert set(altas.iloc[0][["columna_a", "columna_b"]]) == {"a", "b"}


# --------------------------------------------------------------------------- #
def test_auto_clean(df_sucio: pd.DataFrame) -> None:
    limpio, informe = pa.auto_clean(df_sucio)
    assert limpio.shape[0] == len(df_sucio) - 8  # duplicados fuera
    assert limpio.isna().sum().sum() == 0
    assert len(informe) >= 5
    assert all(" " not in c for c in limpio.columns)
    assert pd.api.types.is_datetime64_any_dtype(limpio["fecha_venta"])


def test_smart_cast() -> None:
    serie = pd.Series(["1.234,56", "2.000,00", "3.500,10"])
    resultado = pa.smart_cast(serie)
    assert pd.api.types.is_numeric_dtype(resultado)
    assert resultado.iloc[0] == pytest.approx(1234.56)

    porcentajes = pa.smart_cast(pd.Series(["10%", "25%", "50%"]))
    assert porcentajes.iloc[1] == pytest.approx(0.25)

    moneda = pa.smart_cast(pd.Series(["$ 1.000", "$ 2.500", "$ 300"]))
    assert pd.api.types.is_numeric_dtype(moneda)


def test_imputacion_por_grupo() -> None:
    df = pd.DataFrame({"grupo": ["a", "a", "b", "b"], "valor": [10.0, np.nan, 100.0, np.nan]})
    relleno = pa.impute_missing(df, ["valor"], strategy="median", group_by="grupo")
    assert relleno["valor"].tolist() == [10.0, 10.0, 100.0, 100.0]


def test_cap_outliers() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 1000]})
    recortado = pa.cap_outliers(df, ["x"])
    assert recortado["x"].max() < 1000


def test_pipeline_limpieza(df_sucio: pd.DataFrame) -> None:
    pasos = [
        {"funcion": "normalize_column_names"},
        {"funcion": "drop_duplicates"},
        {"funcion": "impute_missing", "strategy": "median"},
    ]
    limpio, informe = pa.clean_pipeline(df_sucio, pasos)
    assert len(informe) == 3
    assert limpio.shape[0] == len(df_sucio) - 8


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("metodo", ["minmax", "zscore", "robust", "maxabs", "decimal", "log1p", "quantile", "rank"])
def test_metodos_de_escalado(metodo: str) -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [10.0, 20.0, 30.0, 40.0, 50.0]})
    resultado = pa.normalize(df, ["x"], method=metodo)
    assert f"x_{metodo}" in resultado.columns
    assert resultado[f"x_{metodo}"].notna().all()


def test_minmax_rango() -> None:
    df = pd.DataFrame({"x": [0.0, 5.0, 10.0]})
    resultado = pa.normalize(df, ["x"], method="minmax")
    assert resultado["x_minmax"].min() == pytest.approx(0.0)
    assert resultado["x_minmax"].max() == pytest.approx(1.0)


def test_normalizer_fit_transform() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    test = pd.DataFrame({"x": [5.0, 6.0]})
    esc = pa.Normalizer("minmax", ["x"]).fit(train)
    salida_test = esc.transform(test)
    assert salida_test["x"].min() >= 1.0  # valores fuera del rango de entrenamiento
    vuelta = esc.inverse_transform(salida_test)
    assert vuelta["x"].tolist() == pytest.approx([5.0, 6.0])


def test_split_scale_sin_fuga() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"x": [4.0, 5.0]})
    train_esc, test_esc, esc = pa.split_scale(train, test, ["x"], method="zscore")
    assert train_esc["x"].mean() == pytest.approx(0.0, abs=1e-9)
    assert test_esc["x"].mean() > 0  # usa la media de train


def test_encode_categorical() -> None:
    df = pd.DataFrame({"color": ["rojo", "verde", "rojo", "azul"], "precio": [1, 2, 3, 4]})
    onehot = pa.encode_categorical(df, ["color"], method="onehot")
    assert any(c.startswith("color_") for c in onehot.columns)
    frecuencia = pa.encode_categorical(df, ["color"], method="frequency")
    assert "color_frec" in frecuencia.columns
    ordinal = pa.encode_categorical(df, ["color"], method="ordinal")
    assert "color_cod" in ordinal.columns


def test_encode_target() -> None:
    df = pd.DataFrame({"ciudad": ["a", "a", "b", "b"], "y": [1.0, 3.0, 10.0, 14.0]})
    resultado = pa.encode_categorical(df, ["ciudad"], method="target", target="y")
    assert resultado["ciudad_target"].tolist() == pytest.approx([2.0, 2.0, 12.0, 12.0])


def test_discretize() -> None:
    df = pd.DataFrame({"x": list(range(100))})
    resultado = pa.discretize(df, ["x"], bins=4, strategy="quantile")
    assert "x_bin" in resultado.columns
    assert resultado["x_bin"].nunique() == 4


def test_recommend_method() -> None:
    assert pa.recommend_method(pd.Series([1, 2, 3, 4])) in pa.SCALE_METHODS
    assert pa.recommend_method(pd.Series([1, 1, 1, 1000])) == "log1p"


# --------------------------------------------------------------------------- #
def test_features_de_fecha() -> None:
    df = pd.DataFrame({"fecha": pd.date_range("2024-01-01", periods=40, freq="D"), "y": range(40)})
    resultado = pa.add_date_features(df, "fecha")
    assert {"fecha_year", "fecha_month", "fecha_is_weekend", "fecha_mes_sin"} <= set(resultado.columns)


def test_interacciones_y_polinomios() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    assert "a_x_b" in pa.add_interactions(df, ["a", "b"]).columns
    assert "a_menos_b" in pa.add_interactions(df, ["a", "b"], operation="difference").columns
    poly = pa.add_polynomial(df, ["a"], degree=2)
    assert "a_pow2" in poly.columns


def test_rezagos_y_medias_moviles() -> None:
    df = pd.DataFrame({"y": list(range(10))})
    con_lag = pa.add_lag_features(df, ["y"], lags=[1, 2])
    assert con_lag["y_lag1"].iloc[3] == 2
    con_roll = pa.add_rolling_features(df, ["y"], windows=[3])
    assert "y_roll3_mean" in con_roll.columns


def test_agregados_por_grupo() -> None:
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1.0, 3.0, 10.0]})
    resultado = pa.add_aggregate_features(df, "g", ["v"], aggs=["mean"])
    assert resultado["grp_v_mean"].tolist() == [2.0, 2.0, 10.0]


def test_preprocesador_sklearn() -> None:
    df = pd.DataFrame({"num": [1.0, 2.0, np.nan, 4.0], "cat": ["a", "b", "a", "b"], "y": [1, 0, 1, 0]})
    pre = pa.build_preprocessor(df.drop(columns=["y"]))
    matriz = pre.fit_transform(df.drop(columns=["y"]))
    assert matriz.shape[0] == 4
    assert not np.isnan(matriz).any()


def test_seleccion_de_variables() -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    df["c"] = df["a"] * 2 + 0.001
    assert "c" not in pa.select_by_correlation(df, ["a", "b", "c"], threshold=0.9)
    importantes = pa.select_by_importance(df, pd.Series(rng.normal(size=100)), k=2)
    assert len(importantes) == 3


# --------------------------------------------------------------------------- #
def test_graficos_devuelven_figuras(df_sucio: pd.DataFrame) -> None:
    import matplotlib.figure

    df = df_sucio.dropna()
    assert isinstance(pa.histogram(df, "Precio Unitario"), matplotlib.figure.Figure)
    assert isinstance(pa.boxplot(df, "Precio Unitario"), matplotlib.figure.Figure)
    assert isinstance(pa.scatter(df, "Cantidad", "Precio Unitario"), matplotlib.figure.Figure)
    assert isinstance(pa.heatmap_corr(df), matplotlib.figure.Figure)
    assert isinstance(pa.countplot(df, "Categoria"), matplotlib.figure.Figure)
    assert isinstance(pa.missing_matrix(df_sucio), matplotlib.figure.Figure)


def test_auto_plots_y_guardado(tmp_path, df_sucio: pd.DataFrame) -> None:
    figuras = pa.auto_plots(df_sucio)
    assert len(figuras) > 0
    ruta = pa.save_figure(next(iter(figuras.values())), tmp_path / "g.png")
    assert ruta.exists() and ruta.stat().st_size > 0


# --------------------------------------------------------------------------- #
@pytest.fixture()
def serie_temporal() -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=72, freq="MS")
    tendencia = np.linspace(100, 200, 72)
    estacional = 20 * np.sin(2 * np.pi * np.arange(72) / 12)
    ruido = np.random.default_rng(5).normal(0, 4, 72)
    return pd.DataFrame({"fecha": idx, "valor": tendencia + estacional + ruido})


def test_to_datetime_index(serie_temporal: pd.DataFrame) -> None:
    serie = pa.to_datetime_index(serie_temporal, "fecha", "valor", freq="MS", agg="sum")
    assert isinstance(serie.index, pd.DatetimeIndex)
    assert len(serie) == 72


def test_estacionariedad(serie_temporal: pd.DataFrame) -> None:
    serie = pa.to_datetime_index(serie_temporal, "fecha", "valor")["valor"]
    pruebas = pa.stationarity_tests(serie)
    assert "adf_pvalor" in pruebas
    assert pruebas["veredicto"] in {"estacionaria", "no estacionaria", "resultado mixto (revisar)"}


def test_descomposicion(serie_temporal: pd.DataFrame) -> None:
    import matplotlib.figure

    serie = pa.to_datetime_index(serie_temporal, "fecha", "valor")["valor"]
    assert isinstance(pa.decompose(serie, period=12), matplotlib.figure.Figure)
    fuerza = pa.seasonal_strength(serie, period=12)
    assert fuerza["estacionalidad"] > 0.3


@pytest.mark.parametrize("metodo", ["naive", "seasonal_naive", "mean", "drift", "ses", "holt", "holt_winters"])
def test_pronostico(serie_temporal: pd.DataFrame, metodo: str) -> None:
    serie = pa.to_datetime_index(serie_temporal, "fecha", "valor")["valor"]
    pred = pa.forecast(serie, horizon=6, method=metodo, seasonality=12)
    assert len(pred) == 6
    assert pred["pronostico"].notna().all()


def test_pronostico_con_indice_sin_frecuencia(serie_temporal: pd.DataFrame) -> None:
    """Al eliminar filas pandas pierde la frecuencia; el pronostico debe seguir funcionando."""
    serie = pa.to_datetime_index(serie_temporal, "fecha", "valor", freq="MS", agg="mean")["valor"]
    serie = serie.drop(serie.index[5])
    assert serie.index.freq is None
    pred = pa.forecast(serie, horizon=6, method="holt_winters", seasonality=12)
    assert len(pred) == 6
    assert pred["pronostico"].notna().all()


def test_metricas() -> None:
    resultado = pa.metrics([1, 2, 3, 4], [1.1, 1.9, 3.2, 3.8])
    assert resultado["mae"] == pytest.approx(0.15, abs=0.01)
    assert resultado["r2"] > 0.97


def test_backtest(serie_temporal: pd.DataFrame) -> None:
    serie = pa.to_datetime_index(serie_temporal, "fecha", "valor")["valor"]
    tabla = pa.backtest(serie, horizon=6, n_splits=2, method="holt", seasonality=12)
    assert not tabla.empty
    assert "mae" in tabla.columns


# --------------------------------------------------------------------------- #
def test_deteccion_de_tarea() -> None:
    assert pa.detect_task(pd.Series([0, 1, 0, 1])) == "classification"
    assert pa.detect_task(pd.Series([1.5, 2.7, 3.1])) == "regression"


def test_train_model_clasificacion() -> None:
    rng = np.random.default_rng(11)
    n = 300
    df = pd.DataFrame(
        {
            "edad": rng.integers(18, 70, n),
            "ingreso": rng.normal(3000, 800, n).round(2),
            "ciudad": rng.choice(["A", "B", "C"], n),
            "abandono": rng.choice([0, 1], n),
        }
    )
    resultado = pa.train_model(df, target="abandono", model="random_forest")
    assert resultado["tarea"] == "classification"
    assert 0 <= resultado["metricas"]["accuracy"] <= 1
    assert len(resultado["y_pred"]) == len(resultado["y_test"])


def test_train_model_regresion() -> None:
    rng = np.random.default_rng(12)
    n = 200
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "categoria": rng.choice(["a", "b"], n), "y": 3 * x + rng.normal(0, 0.5, n)})
    resultado = pa.train_model(df, target="y", model="random_forest")
    assert resultado["tarea"] == "regression"
    assert resultado["metricas"]["r2"] > 0.5


def test_validacion_cruzada_y_comparacion() -> None:
    rng = np.random.default_rng(13)
    n = 200
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "z": rng.normal(size=n), "y": 2 * x + rng.normal(0, 0.3, n)})
    resumen = pa.cross_validate_model(df, "y", model="ridge", cv=3)
    assert resumen["r2"] > 0.5
    tabla = pa.compare_models(df, "y", models=["lineal", "ridge"], cv=3)
    assert set(tabla["modelo"]) == {"lineal", "ridge"}


def test_importancia_y_residuos() -> None:
    rng = np.random.default_rng(14)
    n = 150
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "ruido": rng.normal(size=n), "y": 5 * x + rng.normal(0, 0.2, n)})
    resultado = pa.train_model(df, target="y", model="random_forest")
    importancia = pa.permutation_importance(resultado["pipeline"], resultado["X_test"], resultado["y_test"])
    assert importancia.iloc[0]["caracteristica"] == "x"
    residuos = pa.residuals(resultado["y_test"], resultado["y_pred"])
    assert residuos["residuo"].abs().mean() < 1.0


# --------------------------------------------------------------------------- #
def test_informe_html(tmp_path, df_sucio: pd.DataFrame) -> None:
    ruta = pa.generate_report(df_sucio, tmp_path / "informe.html", title="Prueba")
    assert ruta.exists()
    contenido = ruta.read_text(encoding="utf-8")
    assert "Prueba" in contenido
    assert "Resumen" in contenido
    assert "data:image/png;base64," in contenido


def test_dataframe_to_html() -> None:
    html = pa.dataframe_to_html(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
    assert "<table" in html


# --------------------------------------------------------------------------- #
def test_cli_perfil(tmp_path, df_sucio: pd.DataFrame, capsys) -> None:
    from pyanalytics.cli import main

    ruta = tmp_path / "datos.csv"
    df_sucio.to_csv(ruta, index=False)
    codigo = main(["perfil", str(ruta)])
    assert codigo == 0
    salida = capsys.readouterr().out
    assert "Resumen" in salida


def test_cli_informe_y_normalizar(tmp_path, df_sucio: pd.DataFrame) -> None:
    from pyanalytics.cli import main

    entrada = tmp_path / "datos.csv"
    df_sucio.dropna().to_csv(entrada, index=False)
    assert main(["normalizar", str(entrada), "--metodo", "minmax", "-o", str(tmp_path / "n.csv")]) == 0
    assert (tmp_path / "n.csv").exists()
    assert main(["informe", str(entrada), "-o", str(tmp_path / "i.html"), "--sin-graficos"]) == 0
    assert (tmp_path / "i.html").exists()


# --------------------------------------------------------------------------- #
# datasets de ejemplo (ingenieria ambiental, Colombia)
# --------------------------------------------------------------------------- #
def test_lista_datasets() -> None:
    assert set(pa.lista_datasets()) == {"calidad_agua", "aire_mensual"}
    assert set(pa.DATASETS) == {"calidad_agua", "aire_mensual"}


def test_calidad_agua() -> None:
    df = pa.calidad_agua(n=120, semilla=1)
    assert df.shape[0] > 120  # incluye duplicados
    assert "Indice ICA" in df.columns
    assert df["Indice ICA"].between(0, 100).all()
    assert set(df["Cumplimiento Norma"].dropna().unique()) <= {"cumple", "no cumple"}
    # suciedad a proposito: nulos, duplicados y numeros como texto espanol
    assert df.isna().any().any()
    assert df.duplicated().any()
    assert pd.api.types.is_string_dtype(df["Solidos Suspendidos (mg/L)"])
    assert "999.0" in df["DBO5 (mg/L)"].astype(str).values


def test_aire_mensual() -> None:
    df = pa.aire_mensual(meses=24, semilla=2)
    assert set(df.columns) == {"fecha", "pm25_ug_m3", "pm10_ug_m3", "estacion"}
    assert df["estacion"].nunique() == 2
    assert df["fecha"].dt.month.eq(1).sum() == 4  # dos estaciones x dos eneros
    assert df["pm10_ug_m3"].mean() > df["pm25_ug_m3"].mean()
    assert df["pm25_ug_m3"].notna().all()
    assert df["pm10_ug_m3"].isna().any()


def test_calidad_agua_tamano_pequeno() -> None:
    df = pa.calidad_agua(n=10, semilla=4)
    assert len(df) == 20  # 10 base + 10 duplicadas
    assert df.isna().any().any()


def test_cargar_ejemplo_desconocido() -> None:
    with pytest.raises(ValueError):
        pa.cargar_ejemplo("no_existe")


def test_datasets_sobreviven_la_limpieza() -> None:
    limpio, _ = pa.auto_clean(pa.calidad_agua(n=100, semilla=3))
    assert "indice_ica" in limpio.columns
    assert pd.api.types.is_numeric_dtype(limpio["indice_ica"])
    assert pd.api.types.is_numeric_dtype(limpio["solidos_suspendidos_mg_l"])
    assert not limpio.duplicated().any()


# --------------------------------------------------------------------------- #
# eda: perfilado interactivo y pipelines automaticos (skrub, opcional)
# --------------------------------------------------------------------------- #
def test_skrub_disponible_devuelve_bool() -> None:
    assert isinstance(pa.skrub_disponible(), bool)
    version = pa.version_skrub()
    assert version is None or isinstance(version, str)


def test_informe_eda_html_es_html() -> None:
    html = pa.informe_eda_html(pa.calidad_agua(n=60, semilla=5), titulo="Calidad del agua")
    assert "<html" in html.lower()
    assert len(html) > 1000


def test_informe_eda_fallback_sin_skrub(monkeypatch) -> None:
    from pyanalytics import eda as eda_mod

    monkeypatch.setattr(eda_mod, "skrub_disponible", lambda: False)
    html = pa.informe_eda_html(pa.aire_mensual(meses=12, semilla=3), titulo="Aire")
    assert "<html" in html.lower()


def test_guardar_informe_eda(tmp_path) -> None:
    ruta = pa.guardar_informe_eda(pa.aire_mensual(meses=12, semilla=6), tmp_path / "sub" / "eda.html")
    assert ruta.exists()
    assert "<html" in ruta.read_text(encoding="utf-8").lower()


def test_asociaciones_columnas() -> None:
    tabla = pa.asociaciones_columnas(pa.calidad_agua(n=80, semilla=8))
    assert isinstance(tabla, pd.DataFrame)
    assert not tabla.empty


def test_asociaciones_columnas_sin_skrub(monkeypatch) -> None:
    from pyanalytics import eda as eda_mod

    monkeypatch.setattr(eda_mod, "skrub_disponible", lambda: False)
    tabla = pa.asociaciones_columnas(pa.aire_mensual(meses=12, semilla=3))
    assert isinstance(tabla, pd.DataFrame)


def test_modelo_rapido_sin_skrub_mensaje(monkeypatch) -> None:
    from pyanalytics import eda as eda_mod

    monkeypatch.setattr(eda_mod, "skrub_disponible", lambda: False)
    with pytest.raises(ImportError):
        pa.modelo_rapido(pa.aire_mensual(meses=6, semilla=1), objetivo="pm25_ug_m3")


@pytest.mark.skipif(not pa.skrub_disponible(), reason="skrub no esta instalado")
def test_modelo_rapido_regresion() -> None:
    res = pa.modelo_rapido(pa.calidad_agua(n=150, semilla=9), objetivo="Indice ICA", semilla=0)
    assert res["tarea"] == "regression"
    assert "r2" in res["metricas"]
    assert len(res["y_pred"]) == res["n_prueba"]
