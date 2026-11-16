"""Genera datasets de ejemplo en la carpeta examples/.

Los datos (sinteticos) son de **ingenieria ambiental en Colombia** y viven en
el modulo :mod:`pyanalytics.datasets`, de modo que la app web y los ejemplos
comparten la misma fuente. Aqui solo se escriben a disco para poder probar la
herramienta con archivos reales.

Ejecutar:
    .venv\\Scripts\\python.exe examples\\generar_datos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from pyanalytics.datasets import aire_mensual, calidad_agua  # noqa: E402

AQUI = Path(__file__).resolve().parent


def main() -> None:
    AQUI.mkdir(parents=True, exist_ok=True)
    agua = calidad_agua()
    aire = aire_mensual()

    agua.to_csv(AQUI / "calidad_agua_sucio.csv", index=False, encoding="utf-8-sig")
    agua.to_excel(AQUI / "calidad_agua_sucio.xlsx", index=False, sheet_name="calidad_agua")
    aire.to_csv(AQUI / "aire_mensual.csv", index=False, encoding="utf-8-sig")

    print(f"calidad_agua_sucio.csv  -> {agua.shape[0]} filas x {agua.shape[1]} columnas")
    print(f"calidad_agua_sucio.xlsx -> {agua.shape[0]} filas x {agua.shape[1]} columnas")
    print(f"aire_mensual.csv        -> {aire.shape[0]} filas x {aire.shape[1]} columnas")


if __name__ == "__main__":
    main()
