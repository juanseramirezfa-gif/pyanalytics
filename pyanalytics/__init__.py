"""pyanalytics - caja de herramientas para analizar, limpiar, normalizar y
visualizar datos con Python.

Modulos
-------
- :mod:`pyanalytics.io`          carga y guardado en multiples formatos
- :mod:`pyanalytics.profile`     exploracion y perfilado
- :mod:`pyanalytics.clean`       limpieza de datos
- :mod:`pyanalytics.normalize`   escalado, codificacion y discretizacion
- :mod:`pyanalytics.features`    ingenieria de caracteristicas
- :mod:`pyanalytics.plots`       graficos estaticos e interactivos
- :mod:`pyanalytics.timeseries`  series temporales y pronostico
- :mod:`pyanalytics.modeling`    modelos, validacion y explicabilidad
- :mod:`pyanalytics.report`      informes HTML
- :mod:`pyanalytics.eda`         perfilado exploratorio interactivo y pipelines (integracion skrub, opcional)
- :mod:`pyanalytics.datasets`    datasets de ejemplo (ingenieria ambiental, Colombia)

Ejemplo rapido
--------------
>>> import pyanalytics as pa
>>> df = pa.load_data("calidad_agua.csv")     # o pa.cargar_ejemplo("calidad_agua")
>>> perfil = pa.full_profile(df)
>>> pa.guardar_informe_eda(df, "outputs/eda.html")   # perfilado interactivo (skrub, opcional)
>>> limpio, informe = pa.auto_clean(df)
>>> pa.save_data(limpio, "limpio.xlsx")
"""

from __future__ import annotations

__version__ = "0.1.0"

from .clean import (
    CleanReport,
    auto_clean,
    cap_outliers,
    clean_pipeline,
    convert_types,
    drop_constant_columns,
    drop_duplicates,
    drop_empty_columns,
    drop_missing_rows,
    impute_missing,
    normalize_column_names,
    remove_outliers,
    replace_values,
    smart_cast,
    strip_whitespace,
)
from .datasets import (
    DATASETS,
    aire_mensual,
    calidad_agua,
    cargar_ejemplo,
    lista_datasets,
)
from .eda import (
    asociaciones_columnas,
    guardar_informe_eda,
    informe_eda_html,
    modelo_rapido,
    skrub_disponible,
    version_skrub,
)
from .features import (
    add_aggregate_features,
    add_date_features,
    add_frequency_features,
    add_interactions,
    add_lag_features,
    add_polynomial,
    add_ratio_features,
    add_rolling_features,
    add_text_features,
    build_preprocessor,
    select_by_correlation,
    select_by_importance,
    select_by_variance,
)
from .io import (
    SUPPORTED_FORMATS,
    describe_formats,
    list_excel_sheets,
    load_data,
    load_folder,
    save_data,
)
from .modeling import (
    compare_models,
    confusion_matrix_df,
    cross_validate_model,
    detect_task,
    evaluate_classification,
    evaluate_regression,
    get_model,
    learning_curve_data,
    list_models,
    permutation_importance,
    residuals,
    split_data,
    train_model,
)
from .normalize import (
    SCALE_METHODS,
    Normalizer,
    discretize,
    encode_categorical,
    normalize,
    recommend_method,
    split_scale,
)
from .plots import (
    auto_plots,
    barplot,
    boxplot,
    countplot,
    figure_to_base64,
    heatmap_corr,
    histogram,
    line,
    missing_matrix,
    pairplot,
    pie,
    scatter,
    save_figure,
    set_style,
    violinplot,
)
from .profile import (
    categorical_columns,
    categorical_stats,
    column_summary,
    column_types,
    correlation_matrix,
    data_quality_score,
    datetime_columns,
    detect_outliers,
    duplicates_report,
    full_profile,
    high_correlations,
    missing_report,
    numeric_columns,
    numeric_stats,
    outliers_report,
    overview,
    value_counts_table,
)
from .report import dataframe_to_html, generate_report
from .timeseries import (
    acf_pacf,
    acf_pacf_plot,
    backtest,
    decompose,
    describe_series,
    difference,
    forecast,
    metrics,
    resample_series,
    rolling_stats,
    seasonal_strength,
    stationarity_tests,
    to_datetime_index,
    train_test_split_ts,
)

__all__ = [
    "__version__",
    # io
    "load_data", "load_folder", "save_data", "list_excel_sheets", "describe_formats", "SUPPORTED_FORMATS",
    # profile
    "full_profile", "overview", "column_summary", "missing_report", "duplicates_report",
    "numeric_stats", "categorical_stats", "correlation_matrix", "high_correlations",
    "detect_outliers", "outliers_report", "value_counts_table", "data_quality_score",
    "numeric_columns", "categorical_columns", "datetime_columns", "column_types",
    # clean
    "CleanReport", "auto_clean", "clean_pipeline", "normalize_column_names", "strip_whitespace",
    "drop_empty_columns", "drop_constant_columns", "drop_duplicates", "drop_missing_rows",
    "impute_missing", "convert_types", "smart_cast", "replace_values", "cap_outliers", "remove_outliers",
    # normalize
    "normalize", "Normalizer", "encode_categorical", "discretize", "recommend_method",
    "split_scale", "SCALE_METHODS",
    # features
    "add_date_features", "add_interactions", "add_polynomial", "add_lag_features",
    "add_rolling_features", "add_aggregate_features", "add_ratio_features",
    "add_text_features", "add_frequency_features", "build_preprocessor",
    "select_by_variance", "select_by_correlation", "select_by_importance",
    # plots
    "set_style", "histogram", "boxplot", "violinplot", "scatter", "line", "barplot",
    "countplot", "pie", "heatmap_corr", "pairplot", "missing_matrix", "auto_plots",
    "save_figure", "figure_to_base64",
    # timeseries
    "to_datetime_index", "resample_series", "rolling_stats", "difference",
    "stationarity_tests", "acf_pacf", "acf_pacf_plot", "decompose", "seasonal_strength",
    "train_test_split_ts", "forecast", "metrics", "backtest", "describe_series",
    # modeling
    "list_models", "get_model", "split_data", "detect_task", "train_model",
    "cross_validate_model", "compare_models", "evaluate_classification",
    "evaluate_regression", "confusion_matrix_df", "permutation_importance",
    "residuals", "learning_curve_data",
    # report
    "generate_report", "dataframe_to_html",
    # eda (skrub, opcional)
    "skrub_disponible", "version_skrub", "informe_eda_html", "guardar_informe_eda",
    "asociaciones_columnas", "modelo_rapido",
    # datasets
    "calidad_agua", "aire_mensual", "DATASETS", "lista_datasets", "cargar_ejemplo",
]
