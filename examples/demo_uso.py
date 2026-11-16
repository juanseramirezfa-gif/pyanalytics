"""Demostracion de extremo a extremo de pyanalytics.

Genera datos de ejemplo (ingenieria ambiental en Colombia), los limpia, los
explora, los normaliza, grafica, analiza una serie temporal, entrena un modelo
y produce un informe HTML.

Ejecutar:
    .venv\\Scripts\\python.exe examples\\demo_uso.py
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import pyanalytics as pa  # noqa: E402

from pyanalytics.datasets import aire_mensual, calidad_agua  # noqa: E402

SALIDA = RAIZ / "outputs"
SALIDA.mkdir(exist_ok=True)

OBJETIVO = "indice_ica"  # columna objetivo para el modelado (regresion)


def separador(titulo: str) -> None:
    print(f"\n{'=' * 70}\n{titulo}\n{'=' * 70}")


def main() -> None:
    # ------------------------------------------------------------------ #
    separador("1. CARGA")
    df = calidad_agua()
    print(f"Dataset crudo: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(df.head(3).to_string(index=False))

    # ------------------------------------------------------------------ #
    separador("2. EXPLORACION")
    perfil = pa.full_profile(df)
    for clave, valor in perfil["overview"].items():
        print(f"  {clave}: {valor}")
    print(f"\n  Calidad: {perfil['calidad']['puntaje']}/100")
    for problema in perfil["calidad"]["problemas"]:
        print(f"    - {problema}")

    print("\n  Columnas con mas nulos:")
    print(perfil["nulos"].head(4).to_string(index=False))

    # ------------------------------------------------------------------ #
    separador("3. LIMPIEZA")
    limpio, informe = pa.auto_clean(df)
    print(informe.summary())
    print(f"\n  Antes:   {df.shape[0]} x {df.shape[1]}")
    print(f"  Despues: {limpio.shape[0]} x {limpio.shape[1]}")
    print(f"  Nulos restantes: {limpio.isna().sum().sum()}")

    # ------------------------------------------------------------------ #
    separador("4. NORMALIZACION")
    numericas = pa.numeric_columns(limpio)
    print("  Recomendacion automatica por columna:")
    for col in numericas:
        print(f"    {col}: {pa.recommend_method(limpio[col])}")
    escalado = pa.normalize(limpio, numericas, method="robust")
    nuevas = [c for c in escalado.columns if c.endswith("_robust")]
    print(f"\n  Columnas escaladas (robust): {len(nuevas)}")
    print(escalado[nuevas].describe().loc[["mean", "std"]].round(3).to_string())

    # ------------------------------------------------------------------ #
    separador("5. GRAFICOS")
    for nombre, figura in list(pa.auto_plots(limpio, target=OBJETIVO).items())[:6]:
        ruta = pa.save_figure(figura, SALIDA / "graficos" / f"{nombre}.png")
        print(f"  {ruta.relative_to(RAIZ)}")

    # ------------------------------------------------------------------ #
    separador("6. SERIES TEMPORALES")
    aire = aire_mensual()
    estacion = aire[aire["estacion"] == "Estacion Centro"]
    serie = pa.to_datetime_index(estacion, "fecha", "pm25_ug_m3", freq="MS", agg="mean")["pm25_ug_m3"]
    print(f"  Serie mensual: {len(serie)} observaciones ({serie.index[0].date()} -> {serie.index[-1].date()})")
    info = pa.describe_series(serie)
    print(f"  Estacionariedad: {info.get('veredicto')} (ADF p={info.get('adf_pvalor')})")
    print(f"  Fuerza estacional: {info.get('estacionalidad')}")
    pa.save_figure(pa.decompose(serie, period=12), SALIDA / "graficos" / "descomposicion.png")

    pred = pa.forecast(serie, horizon=12, method="holt_winters", seasonality=12)
    print("\n  Pronostico (12 meses):")
    print(pred.round(2).to_string())
    print(f"  MAPE del pronostico: {pa.metrics(serie[-12:], pred['pronostico'][:12])['mape']}%")

    # ------------------------------------------------------------------ #
    separador("7. MODELADO")
    tabla = pa.compare_models(limpio, target=OBJETIVO, task="regression", models=["lineal", "ridge", "random_forest"], cv=3)
    print(tabla.to_string(index=False))

    resultado = pa.train_model(limpio, target=OBJETIVO, model="random_forest")
    print("\n  Metricas en prueba:")
    for clave, valor in resultado["metricas"].items():
        print(f"    {clave}: {valor}")

    importancia = pa.permutation_importance(resultado["pipeline"], resultado["X_test"], resultado["y_test"])
    print("\n  Variables mas importantes:")
    print(importancia.head(5).to_string(index=False))

    # ------------------------------------------------------------------ #
    separador("8. EXPORTAR")
    print(f"  {pa.save_data(limpio, SALIDA / 'calidad_agua_limpio.xlsx')}")
    print(f"  {pa.save_data(escalado, SALIDA / 'calidad_agua_escalado.csv')}")
    print(f"  {pa.generate_report(limpio, SALIDA / 'informe.html', title='Calidad del agua - Colombia')}")
    print(f"  {pa.guardar_informe_eda(limpio, SALIDA / 'perfil_exploratorio.html', titulo='Calidad del agua - Colombia')}")
    print(f"  (perfilado interactivo con skrub: {'si' if pa.skrub_disponible() else 'no instalado, se uso el informe propio'})")

    print("\nDemostracion completada.")


if __name__ == "__main__":
    main()
