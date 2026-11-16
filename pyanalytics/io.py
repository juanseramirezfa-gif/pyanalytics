"""Carga y guardado de datos en multiples formatos.

Soporta archivos locales, carpetas y URLs. El formato se deduce de la
extension del archivo, pero siempre puede forzarse con ``kind=``.

Ejemplos
--------
>>> df = load_data("calidad_agua.csv")
>>> df = load_data("libro.xlsx", sheet_name="Hoja1")
>>> df = load_data("datos.parquet")
>>> save_data(df, "limpio.xlsx", sheet_name="limpio")
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "SUPPORTED_FORMATS",
    "load_data",
    "load_folder",
    "save_data",
    "list_excel_sheets",
    "describe_formats",
]

#: Formatos reconocidos por extension.
SUPPORTED_FORMATS: dict[str, str] = {
    ".csv": "CSV delimitado por comas",
    ".txt": "Texto delimitado (usa sep=)",
    ".tsv": "Texto delimitado por tabuladores",
    ".xlsx": "Excel moderno (openpyxl)",
    ".xlsm": "Excel con macros",
    ".xls": "Excel antiguo",
    ".xlsb": "Excel binario",
    ".ods": "OpenDocument Spreadsheet",
    ".json": "JSON (records, columns, split o lines)",
    ".jsonl": "JSON Lines",
    ".ndjson": "JSON Lines",
    ".parquet": "Apache Parquet (columnar)",
    ".feather": "Arrow Feather",
    ".orc": "Apache ORC",
    ".pkl": "Pickle de pandas",
    ".pickle": "Pickle de pandas",
    ".dta": "Stata",
    ".sav": "SPSS",
    ".sas7bdat": "SAS",
    ".html": "Tablas HTML",
    ".htm": "Tablas HTML",
    ".xml": "XML",
    ".db": "SQLite",
    ".sqlite": "SQLite",
    ".sqlite3": "SQLite",
    ".zip": "Archivo comprimido con un unico tabular dentro",
}

_TEXT_FORMATS = {".csv", ".txt", ".tsv"}
_EXCEL_FORMATS = {".xlsx", ".xlsm", ".xls", ".xlsb", ".ods"}
_JSON_FORMATS = {".json", ".jsonl", ".ndjson"}
_SQLITE_FORMATS = {".db", ".sqlite", ".sqlite3"}


def _suffix(path: str | Path) -> str:
    return Path(str(path)).suffix.lower()


def _is_url(source: str) -> bool:
    return str(source).lower().startswith(("http://", "https://", "ftp://"))


def _suffix_from_url(url: str) -> str:
    clean = url.split("?")[0].split("#")[0]
    return Path(clean).suffix.lower()


def list_excel_sheets(path: str | Path) -> list[str]:
    """Devuelve los nombres de las hojas de un libro de Excel."""
    with pd.ExcelFile(path) as xls:
        return list(xls.sheet_names)


def _read_text(source: Any, sep: str, encoding: str, **kwargs: Any) -> pd.DataFrame:
    if _is_url(source):
        return pd.read_csv(source, sep=sep, encoding=encoding, **kwargs)
    return pd.read_csv(source, sep=sep, encoding=encoding, **kwargs)


def load_data(
    source: str | Path,
    *,
    kind: str | None = None,
    sheet_name: str | int | None = 0,
    sep: str = ",",
    encoding: str = "utf-8",
    json_orient: str | None = None,
    lines: bool = False,
    table_index: int = 0,
    query: str | None = None,
    connection: Any = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Carga un archivo (o URL) y devuelve un :class:`pandas.DataFrame`.

    Parametros
    ----------
    source:
        Ruta local, carpeta no (usa :func:`load_folder`) o URL.
    kind:
        Fuerza el formato (``"csv"``, ``"excel"``, ``"json"``, ...). Si es
        ``None`` se deduce de la extension.
    sheet_name:
        Hoja de Excel a leer. ``None`` lee todas y devuelve un diccionario.
    sep:
        Separador para CSV/TXT/TSV.
    encoding:
        Codificacion del archivo de texto.
    json_orient:
        Orientacion de ``pd.read_json`` (``"records"``, ``"columns"``, ...).
    lines:
        ``True`` para JSON Lines / NDJSON.
    table_index:
        Indice de la tabla cuando se leen HTML.
    query:
        Consulta SQL cuando la fuente es una base SQLite o ``connection``.
    connection:
        Conexion SQLAlchemy o sqlite3 existente.

    Devuelve
    --------
    pandas.DataFrame
    """
    if connection is not None or kind == "sql":
        if not query:
            raise ValueError("Para leer de una base de datos necesitas 'query'.")
        engine = connection
        if engine is None:
            engine = sqlite3.connect(str(source))
        return pd.read_sql_query(query, engine)

    suffix = _suffix_from_url(str(source)) if _is_url(source) else _suffix(source)
    fmt = (kind or suffix.lstrip(".")).lower()

    if fmt in ("csv", "txt", "tsv") or suffix in _TEXT_FORMATS:
        if fmt == "tsv" or suffix == ".tsv":
            sep = kwargs.pop("sep", "\t")
        return _read_text(source, sep=sep, encoding=encoding, **kwargs)

    if fmt in ("excel", "xlsx", "xls", "xlsm", "xlsb", "ods") or suffix in _EXCEL_FORMATS:
        return pd.read_excel(source, sheet_name=sheet_name, **kwargs)

    if fmt in ("json", "jsonl", "ndjson") or suffix in _JSON_FORMATS:
        if lines or fmt in ("jsonl", "ndjson") or suffix in (".jsonl", ".ndjson"):
            return pd.read_json(source, lines=True, **kwargs)
        orient = json_orient or "records"
        try:
            return pd.read_json(source, orient=orient, **kwargs)
        except ValueError:
            return pd.read_json(source, **kwargs)

    if fmt == "parquet" or suffix == ".parquet":
        return pd.read_parquet(source, **kwargs)

    if fmt in ("feather", "featherv2") or suffix == ".feather":
        return pd.read_feather(source, **kwargs)

    if fmt == "orc" or suffix == ".orc":
        return pd.read_orc(source, **kwargs)

    if fmt in ("pickle", "pkl") or suffix in (".pkl", ".pickle"):
        return pd.read_pickle(source, **kwargs)

    if fmt in ("dta", "stata") or suffix == ".dta":
        return pd.read_stata(source, **kwargs)

    if fmt in ("sav", "spss") or suffix == ".sav":
        return pd.read_spss(source, **kwargs)

    if fmt == "sas7bdat" or suffix == ".sas7bdat":
        return pd.read_sas(source, **kwargs)

    if fmt in ("html", "htm") or suffix in (".html", ".htm"):
        tables = pd.read_html(source, **kwargs)
        if not tables:
            raise ValueError("No se encontro ninguna tabla en el HTML.")
        return tables[table_index]

    if fmt == "xml" or suffix == ".xml":
        return pd.read_xml(source, **kwargs)

    if fmt in ("sqlite", "db", "sqlite3") or suffix in _SQLITE_FORMATS:
        if not query:
            raise ValueError("Para leer SQLite indica 'query' con el SELECT.")
        with sqlite3.connect(str(source)) as conn:
            return pd.read_sql_query(query, conn)

    if fmt == "zip" or suffix == ".zip":
        import zipfile

        with zipfile.ZipFile(source) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            if not names:
                raise ValueError("El ZIP esta vacio.")
            inner = names[0]
            with zf.open(inner) as fh:
                data = fh.read()
        import io as _io

        buffer = _io.BytesIO(data)
        return load_data(buffer, kind=_suffix(inner).lstrip("."), sep=sep, **kwargs)

    raise ValueError(
        f"Formato no soportado: {suffix or fmt!r}. "
        f"Formatos validos: {', '.join(sorted(SUPPORTED_FORMATS))}"
    )


def load_folder(
    folder: str | Path,
    *,
    pattern: str = "*",
    recursive: bool = False,
    ignore_errors: bool = False,
    add_source_column: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Concatena todos los tabulares de una carpeta en un unico DataFrame.

    Anade la columna ``_source_file`` con el archivo de origen de cada fila.
    """
    folder = Path(folder)
    files = sorted(folder.rglob(pattern) if recursive else folder.glob(pattern))
    frames: list[pd.DataFrame] = []
    for file in files:
        if file.suffix.lower() not in SUPPORTED_FORMATS or file.suffix.lower() == ".zip":
            continue
        try:
            part = load_data(file, **kwargs)
        except Exception:
            if ignore_errors:
                continue
            raise
        if add_source_column:
            part = part.assign(_source_file=file.name)
        frames.append(part)
    if not frames:
        raise FileNotFoundError(f"No se encontro ningun tabular en {folder}")
    return pd.concat(frames, ignore_index=True)


def save_data(
    df: pd.DataFrame,
    path: str | Path,
    *,
    sheet_name: str = "datos",
    index: bool = False,
    encoding: str = "utf-8-sig",
    **kwargs: Any,
) -> Path:
    """Guarda un DataFrame y devuelve la ruta final.

    Crea los directorios intermedios si no existen.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = _suffix(path)

    if suffix in _TEXT_FORMATS:
        sep = kwargs.pop("sep", "\t" if suffix == ".tsv" else ",")
        df.to_csv(path, sep=sep, index=index, encoding=encoding, **kwargs)
    elif suffix in _EXCEL_FORMATS:
        df.to_excel(path, sheet_name=sheet_name, index=index, **kwargs)
    elif suffix in (".parquet",):
        df.to_parquet(path, index=index, **kwargs)
    elif suffix == ".feather":
        df.reset_index(drop=not index).to_feather(path, **kwargs)
    elif suffix == ".orc":
        df.to_orc(path, index=index, **kwargs)
    elif suffix in (".pkl", ".pickle"):
        df.to_pickle(path, **kwargs)
    elif suffix in _JSON_FORMATS:
        lines = suffix in (".jsonl", ".ndjson")
        df.to_json(path, orient="records", lines=lines, force_ascii=False, **kwargs)
    elif suffix == ".dta":
        df.to_stata(path, write_index=index, **kwargs)
    elif suffix in _SQLITE_FORMATS:
        table = kwargs.pop("table", "datos")
        with sqlite3.connect(str(path)) as conn:
            df.to_sql(table, conn, if_exists=kwargs.pop("if_exists", "replace"), index=index)
    elif suffix in (".html", ".htm"):
        df.to_html(path, index=index, **kwargs)
    elif suffix == ".xml":
        df.to_xml(path, index=index, **kwargs)
    else:
        raise ValueError(
            f"No se puede guardar en {suffix!r}. "
            f"Formatos validos: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )
    return path


def describe_formats() -> pd.DataFrame:
    """Devuelve una tabla con los formatos soportados."""
    rows = [{"extension": ext, "descripcion": desc} for ext, desc in SUPPORTED_FORMATS.items()]
    return pd.DataFrame(rows)
