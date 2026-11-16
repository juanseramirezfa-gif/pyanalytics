"""Interfaz de linea de comandos de pyanalytics.

Ejemplos
--------
    python -m pyanalytics.cli perfil datos.csv
    python -m pyanalytics.cli limpiar datos.csv -o limpio.xlsx
    python -m pyanalytics.cli normalizar datos.csv --metodo minmax -o norm.csv
    python -m pyanalytics.cli informe datos.csv -o informe.html
    python -m pyanalytics.cli graficos datos.csv -d outputs/graficos
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .clean import auto_clean
from .io import load_data, save_data
from .normalize import SCALE_METHODS, encode_categorical, normalize
from .plots import auto_plots, save_figure
from .profile import full_profile
from .report import generate_report


def _cargar(args: argparse.Namespace) -> pd.DataFrame:
    kwargs: dict[str, Any] = {}
    if args.hoja is not None:
        kwargs["sheet_name"] = args.hoja
    if args.sep:
        kwargs["sep"] = args.sep
    return load_data(args.entrada, **kwargs)


def _imprimir_tabla(df: pd.DataFrame, titulo: str) -> None:
    print(f"\n=== {titulo} ===")
    if df is None or len(df) == 0:
        print("(sin datos)")
        return
    with pd.option_context("display.max_rows", 40, "display.max_columns", 30, "display.width", 160):
        print(df.to_string(index=False))


def cmd_perfil(args: argparse.Namespace) -> int:
    df = _cargar(args)
    perfil = full_profile(df)
    ov = perfil["overview"]
    print("=== Resumen ===")
    for clave, valor in ov.items():
        print(f"  {clave}: {valor}")
    _imprimir_tabla(perfil["columnas"], "Columnas")
    _imprimir_tabla(perfil["nulos"].head(20), "Nulos")
    if not perfil["estadisticos_numericos"].empty:
        _imprimir_tabla(perfil["estadisticos_numericos"].reset_index().rename(columns={"index": "columna"}), "Estadisticos numericos")
    if not perfil["correlaciones_altas"].empty:
        _imprimir_tabla(perfil["correlaciones_altas"], "Correlaciones altas")
    _imprimir_tabla(perfil["atipicos"], "Atipicos")
    print("\n=== Calidad ===")
    print(f"  puntaje: {perfil['calidad']['puntaje']}/100")
    for problema in perfil["calidad"]["problemas"]:
        print(f"  - {problema}")
    if args.json:
        salida = {
            "overview": ov,
            "calidad": perfil["calidad"],
            "columnas": perfil["columnas"].to_dict(orient="records"),
        }
        Path(args.json).write_text(json.dumps(salida, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"\nJSON guardado en {args.json}")
    return 0


def cmd_limpiar(args: argparse.Namespace) -> int:
    df = _cargar(args)
    limpio, informe = auto_clean(df, missing_threshold=args.umbral_nulos)
    print("=== Limpieza aplicada ===")
    print(informe.summary())
    print(f"\nAntes: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"Despues: {limpio.shape[0]} filas x {limpio.shape[1]} columnas")
    destino = args.salida or "limpio.csv"
    ruta = save_data(limpio, destino)
    print(f"Guardado en {ruta}")
    if args.informe:
        print(f"Informe de limpieza: {save_data(informe.to_frame(), args.informe)}")
    return 0


def cmd_normalizar(args: argparse.Namespace) -> int:
    df = _cargar(args)
    columnas = args.columnas.split(",") if args.columnas else None
    resultado = normalize(df, columnas, method=args.metodo, keep_original=not args.reemplazar)
    if args.codificar:
        resultado = encode_categorical(resultado, method=args.codificar)
    destino = args.salida or f"normalizado_{args.metodo}.csv"
    print(f"Metodo: {args.metodo}")
    print(f"Nuevas columnas: {[c for c in resultado.columns if c not in df.columns]}")
    print(f"Guardado en {save_data(resultado, destino)}")
    return 0


def cmd_graficos(args: argparse.Namespace) -> int:
    df = _cargar(args)
    carpeta = Path(args.directorio or "outputs/graficos")
    carpeta.mkdir(parents=True, exist_ok=True)
    figuras = auto_plots(df, target=args.objetivo)
    for nombre, fig in figuras.items():
        ruta = save_figure(fig, carpeta / f"{nombre}.png", dpi=args.dpi)
        print(f"  {ruta}")
    print(f"{len(figuras)} grafico(s) en {carpeta}")
    return 0


def cmd_informe(args: argparse.Namespace) -> int:
    df = _cargar(args)
    ruta = generate_report(
        df,
        args.salida or "outputs/informe.html",
        title=args.titulo,
        include_plots=not args.sin_graficos,
    )
    print(f"Informe generado: {ruta}")
    return 0


def cmd_convertir(args: argparse.Namespace) -> int:
    df = _cargar(args)
    ruta = save_data(df, args.salida)
    print(f"{df.shape[0]} filas x {df.shape[1]} columnas -> {ruta}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyanalytics",
        description="Herramienta de analisis, limpieza, normalizacion y visualizacion de datos.",
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    def comun(p: argparse.ArgumentParser, salida: bool = False) -> None:
        p.add_argument("entrada", help="Archivo de datos (csv, xlsx, parquet, json...)")
        p.add_argument("--hoja", default=None, help="Hoja de Excel a leer")
        p.add_argument("--sep", default=None, help="Separador para CSV")
        if salida:
            p.add_argument("-o", "--salida", default=None, help="Archivo de salida")

    p_perfil = sub.add_parser("perfil", help="Resumen y diagnostico del dataset")
    comun(p_perfil)
    p_perfil.add_argument("--json", default=None, help="Guardar el perfil en JSON")
    p_perfil.set_defaults(func=cmd_perfil)

    p_limpiar = sub.add_parser("limpiar", help="Limpieza automatica con trazabilidad")
    comun(p_limpiar, salida=True)
    p_limpiar.add_argument("--umbral-nulos", type=float, default=0.6, help="Elimina columnas con mas nulos que este umbral (0-1)")
    p_limpiar.add_argument("--informe", default=None, help="Guardar el informe de limpieza")
    p_limpiar.set_defaults(func=cmd_limpiar)

    p_norm = sub.add_parser("normalizar", help="Escalar/normalizar columnas")
    comun(p_norm, salida=True)
    p_norm.add_argument("--metodo", default="zscore", choices=list(SCALE_METHODS))
    p_norm.add_argument("--columnas", default=None, help="Lista separada por comas")
    p_norm.add_argument("--codificar", default=None, choices=["onehot", "ordinal", "label", "frequency", "target"])
    p_norm.add_argument("--reemplazar", action="store_true", help="Sobrescribe las columnas originales")
    p_norm.set_defaults(func=cmd_normalizar)

    p_graf = sub.add_parser("graficos", help="Generar graficos automaticos")
    comun(p_graf)
    p_graf.add_argument("-d", "--directorio", default="outputs/graficos")
    p_graf.add_argument("--objetivo", default=None, help="Columna objetivo para graficos dirigidos")
    p_graf.add_argument("--dpi", type=int, default=150)
    p_graf.set_defaults(func=cmd_graficos)

    p_inf = sub.add_parser("informe", help="Generar informe HTML")
    comun(p_inf, salida=True)
    p_inf.add_argument("--titulo", default="Informe de datos")
    p_inf.add_argument("--sin-graficos", action="store_true")
    p_inf.set_defaults(func=cmd_informe)

    p_conv = sub.add_parser("convertir", help="Convertir entre formatos")
    comun(p_conv)
    p_conv.add_argument("salida", help="Archivo de destino")
    p_conv.set_defaults(func=cmd_convertir)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"Error: no se encontro el archivo. {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
