"""Generacion de informes HTML con perfilado y graficos incrustados.

Ejemplos
--------
>>> from pyanalytics.report import generate_report
>>> generate_report(df, "outputs/informe.html", title="Calidad del agua - Colombia")
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .profile import full_profile

__all__ = ["generate_report", "dataframe_to_html", "report_sections"]

_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ --acento: #2563eb; --texto: #1f2937; --suave: #6b7280; --borde: #e5e7eb; --fondo: #f8fafc; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0;
         color: var(--texto); background: var(--fondo); line-height: 1.55; }}
  header {{ background: linear-gradient(135deg, #1e3a8a, #2563eb); color: #fff; padding: 32px 40px; }}
  header h1 {{ margin: 0 0 6px; font-size: 26px; }}
  header p {{ margin: 0; opacity: .9; font-size: 14px; }}
  main {{ max-width: 1200px; margin: 0 auto; padding: 28px 20px 80px; }}
  section {{ background: #fff; border: 1px solid var(--borde); border-radius: 12px; padding: 22px 24px;
             margin-bottom: 22px; box-shadow: 0 1px 2px rgba(15,23,42,.04); }}
  h2 {{ font-size: 18px; margin: 0 0 14px; padding-bottom: 8px; border-bottom: 2px solid var(--borde); }}
  h3 {{ font-size: 15px; margin: 18px 0 8px; color: #374151; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; overflow-x: auto; display: block; }}
  th, td {{ border: 1px solid var(--borde); padding: 6px 10px; text-align: left; white-space: nowrap; }}
  th {{ background: #f1f5f9; font-weight: 600; position: sticky; top: 0; }}
  tr:nth-child(even) td {{ background: #fafafa; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; }}
  .kpi {{ background: var(--fondo); border: 1px solid var(--borde); border-radius: 10px; padding: 14px; }}
  .kpi .valor {{ font-size: 22px; font-weight: 700; color: var(--acento); }}
  .kpi .etiqueta {{ font-size: 12px; color: var(--suave); text-transform: uppercase; letter-spacing: .4px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }}
  .grafico {{ border: 1px solid var(--borde); border-radius: 10px; padding: 10px; background: #fff; }}
  .grafico img {{ width: 100%; height: auto; display: block; }}
  .aviso {{ background: #fffbeb; border-left: 4px solid #f59e0b; padding: 12px 16px; border-radius: 6px; font-size: 14px; }}
  ul.problemas {{ margin: 8px 0 0 18px; padding: 0; }}
  footer {{ text-align: center; color: var(--suave); font-size: 12px; padding: 20px; }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <p>Generado el {fecha} &middot; pyanalytics</p>
</header>
<main>
{contenido}
</main>
<footer>Informe generado automaticamente por pyanalytics</footer>
</body>
</html>
"""


def dataframe_to_html(df: pd.DataFrame, max_rows: int = 50, clases: str = "tabla") -> str:
    """Convierte un DataFrame en una tabla HTML (limitando filas)."""
    if df is None or df.empty:
        return '<p class="aviso">Sin datos para mostrar.</p>'
    vista = df.head(max_rows).copy()
    if len(df) > max_rows:
        resumen = pd.DataFrame([["..."] * len(vista.columns)], columns=vista.columns)
        vista = pd.concat([vista, resumen], ignore_index=True)
    return vista.to_html(classes=clases, border=0, escape=True, index=False, na_rep="")


def report_sections(df: pd.DataFrame, perfil: dict[str, Any] | None = None) -> dict[str, str]:
    """Construye el HTML de cada seccion del informe."""
    perfil = perfil or full_profile(df)
    ov = perfil["overview"]
    calidad = perfil["calidad"]

    kpis = {
        "Filas": f"{ov['filas']:,}",
        "Columnas": ov["columnas"],
        "Memoria": ov["memoria_legible"],
        "Nulos": f"{ov['porcentaje_nulos']}%",
        "Duplicados": f"{ov['filas_duplicadas']:,}",
        "Calidad": f"{calidad['puntaje']}/100",
    }
    kpi_html = "".join(
        f'<div class="kpi"><div class="valor">{valor}</div><div class="etiqueta">{nombre}</div></div>'
        for nombre, valor in kpis.items()
    )

    problemas = calidad["problemas"]
    aviso = (
        "<h3>Problemas detectados</h3><ul class='problemas'>"
        + "".join(f"<li>{p}</li>" for p in problemas)
        + "</ul>"
        if problemas
        else "<p>No se detectaron problemas relevantes de calidad.</p>"
    )

    return {
        "resumen": f'<section><h2>Resumen</h2><div class="kpis">{kpi_html}</div>{aviso}</section>',
        "columnas": f'<section><h2>Columnas</h2>{dataframe_to_html(perfil["columnas"], 200)}</section>',
        "nulos": f'<section><h2>Valores nulos</h2>{dataframe_to_html(perfil["nulos"], 200)}</section>',
        "numericos": f'<section><h2>Estadisticos numericos</h2>{dataframe_to_html(perfil["estadisticos_numericos"].reset_index().rename(columns={"index": "columna"}), 200)}</section>',
        "categoricos": f'<section><h2>Variables categoricas</h2>{dataframe_to_html(perfil["estadisticos_categoricos"], 200)}</section>',
        "correlaciones": f'<section><h2>Correlaciones altas</h2>{dataframe_to_html(perfil["correlaciones_altas"], 100)}</section>',
        "atipicos": f'<section><h2>Valores atipicos</h2>{dataframe_to_html(perfil["atipicos"], 200)}</section>',
    }


def generate_report(
    df: pd.DataFrame,
    path: str | Path = "outputs/informe.html",
    *,
    title: str = "Informe de datos",
    include_plots: bool = True,
    max_plots: int = 12,
    max_rows: int = 50,
) -> Path:
    """Genera un informe HTML autocontenido (tablas + graficos) y devuelve la ruta."""
    perfil = full_profile(df)
    secciones = report_sections(df, perfil)

    if include_plots:
        try:
            from .plots import auto_plots, figure_to_base64

            figuras = auto_plots(df)
            bloques: list[str] = []
            for nombre, fig in list(figuras.items())[:max_plots]:
                imagen = figure_to_base64(fig)
                bloques.append(
                    f'<div class="grafico"><img src="data:image/png;base64,{imagen}" alt="{nombre}">'
                    f'<p style="text-align:center;font-size:12px;color:#6b7280">{nombre}</p></div>'
                )
            if bloques:
                secciones["graficos"] = (
                    f'<section><h2>Graficos</h2><div class="grid">{"".join(bloques)}</div></section>'
                )
        except Exception as exc:  # los graficos son opcionales
            secciones["graficos"] = f'<section><h2>Graficos</h2><p class="aviso">No se pudieron generar: {exc}</p></section>'

    contenido = "".join(secciones.values())
    html = _TEMPLATE.format(
        title=title,
        fecha=datetime.now().strftime("%d/%m/%Y %H:%M"),
        contenido=contenido,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
