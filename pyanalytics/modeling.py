"""Modelado: division de datos, entrenamiento, evaluacion y explicabilidad.

Sigue el ciclo de vida descrito en las fuentes del notebook: preprocesamiento
-> entrenamiento -> validacion cruzada -> evaluacion -> interpretacion.

Ejemplos
--------
>>> from pyanalytics.modeling import train_model, compare_models
>>> resultado = train_model(df, target="precio")
>>> resultado["metricas"]
>>> compare_models(df, target="abandono")
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from .features import build_preprocessor

__all__ = [
    "list_models",
    "get_model",
    "split_data",
    "detect_task",
    "train_model",
    "cross_validate_model",
    "compare_models",
    "evaluate_classification",
    "evaluate_regression",
    "confusion_matrix_df",
    "permutation_importance",
    "residuals",
    "learning_curve_data",
]

Task = Literal["auto", "classification", "regression"]

_CLASSIFIERS = {
    "logistica": "LogisticRegression",
    "arbol": "DecisionTreeClassifier",
    "random_forest": "RandomForestClassifier",
    "gradient_boosting": "GradientBoostingClassifier",
    "hist_gradient_boosting": "HistGradientBoostingClassifier",
    "knn": "KNeighborsClassifier",
    "svm": "SVC",
    "naive_bayes": "GaussianNB",
    "extra_trees": "ExtraTreesClassifier",
}

_REGRESSORS = {
    "lineal": "LinearRegression",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "arbol": "DecisionTreeRegressor",
    "random_forest": "RandomForestRegressor",
    "gradient_boosting": "GradientBoostingRegressor",
    "hist_gradient_boosting": "HistGradientBoostingRegressor",
    "knn": "KNeighborsRegressor",
    "svm": "SVR",
    "extra_trees": "ExtraTreesRegressor",
}

_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "random_forest": {"n_estimators": 300, "random_state": 0, "n_jobs": -1},
    "extra_trees": {"n_estimators": 300, "random_state": 0, "n_jobs": -1},
    "gradient_boosting": {"random_state": 0},
    "hist_gradient_boosting": {"random_state": 0},
    "logistica": {"max_iter": 2000},
    "arbol": {"random_state": 0},
    "svm": {"probability": True},
    "knn": {"n_neighbors": 5},
}


def list_models(task: Literal["classification", "regression"]) -> list[str]:
    """Nombres de modelos disponibles para la tarea indicada."""
    return sorted((_CLASSIFIERS if task == "classification" else _REGRESSORS).keys())


def get_model(name: str, task: Literal["classification", "regression"] = "regression", **params: Any):
    """Instancia un modelo de scikit-learn por su nombre corto."""
    import sklearn

    registro = _CLASSIFIERS if task == "classification" else _REGRESSORS
    if name == "auto":
        name = "random_forest"
    if name not in registro:
        raise ValueError(f"Modelo desconocido: {name!r}. Disponibles: {list_models(task)}")
    clase = getattr(sklearn, registro[name], None)
    if clase is None:  # algunos viven en submodulos
        import importlib

        for modulo in ("sklearn.linear_model", "sklearn.ensemble", "sklearn.tree", "sklearn.neighbors", "sklearn.svm", "sklearn.naive_bayes"):
            try:
                clase = getattr(importlib.import_module(modulo), registro[name])
                break
            except AttributeError:
                continue
    if clase is None:
        raise ImportError(f"No se pudo importar el modelo {registro[name]}")
    argumentos = dict(_DEFAULT_PARAMS.get(name, {}))
    argumentos.update(params)
    argumentos = _filtrar_parametros(clase, argumentos)
    return clase(**argumentos)


def _filtrar_parametros(clase: Any, argumentos: dict[str, Any]) -> dict[str, Any]:
    """Descarta argumentos que el estimador no acepta (p. ej. random_state en LinearRegression)."""
    import inspect

    try:
        firma = inspect.signature(clase.__init__)
    except (TypeError, ValueError):  # pragma: no cover
        return argumentos
    acepta_var_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in firma.parameters.values()
    )
    if acepta_var_kwargs:
        return argumentos
    validos = set(firma.parameters) - {"self"}
    return {k: v for k, v in argumentos.items() if k in validos}


def detect_task(y: pd.Series, max_classes: int = 20) -> Literal["classification", "regression"]:
    """Decide si el problema es de clasificacion o de regresion."""
    if pd.api.types.is_bool_dtype(y):
        return "classification"
    if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_integer_dtype(y):
        return "regression"
    if y.nunique(dropna=True) <= max_classes:
        return "classification"
    return "regression"


def split_data(
    df: pd.DataFrame,
    target: str,
    *,
    test_size: float = 0.2,
    val_size: float | None = None,
    stratify: bool = True,
    random_state: int = 0,
    task: Task = "auto",
) -> dict[str, Any]:
    """Divide el DataFrame en entrenamiento / (validacion) / prueba."""
    from sklearn.model_selection import train_test_split

    if target not in df.columns:
        raise KeyError(f"El target {target!r} no existe en el DataFrame.")
    y = df[target]
    X = df.drop(columns=[target])
    tarea = detect_task(y) if task == "auto" else task
    estratos = y if (stratify and tarea == "classification" and y.value_counts().min() >= 2) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=estratos
    )
    resultado: dict[str, Any] = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "tarea": tarea,
        "target": target,
    }
    if val_size:
        estratos_val = y_train if (str(y_train.dtype) == "category" or tarea == "classification") else None
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=random_state, stratify=estratos_val
        )
        resultado.update({"X_train": X_tr, "X_val": X_val, "y_train": y_tr, "y_val": y_val})
    return resultado


def train_model(
    df: pd.DataFrame,
    target: str,
    *,
    model: str = "auto",
    task: Task = "auto",
    test_size: float = 0.2,
    scale_method: str = "zscore",
    max_categories: int | None = 20,
    random_state: int = 0,
    **model_params: Any,
) -> dict[str, Any]:
    """Entrena un modelo con preprocesamiento automatico y lo evalua.

    Devuelve un diccionario con el ``pipeline`` entrenado, las metricas,
    el conjunto de prueba y las predicciones.
    """
    from sklearn.pipeline import Pipeline

    division = split_data(df, target, test_size=test_size, task=task, random_state=random_state)
    tarea = division["tarea"]
    X_train, X_test = division["X_train"], division["X_test"]
    y_train, y_test = division["y_train"], division["y_test"]

    preprocesador = build_preprocessor(
        X_train,
        numeric_strategy="median",
        scale_numeric=True,
        scale_method=scale_method,
        categorical_strategy="onehot",
        max_categories=max_categories,
    )
    estimador = get_model(model, tarea, random_state=random_state, **model_params)
    pipeline = Pipeline([("preprocesar", preprocesador), ("modelo", estimador)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metricas = (
        evaluate_classification(y_test, y_pred, _proba(pipeline, X_test))
        if tarea == "classification"
        else evaluate_regression(y_test, y_pred)
    )
    return {
        "pipeline": pipeline,
        "tarea": tarea,
        "metricas": metricas,
        "y_test": y_test,
        "y_pred": y_pred,
        "X_test": X_test,
        "modelo": model,
        "nombres_caracteristicas": _feature_names(pipeline, X_train),
    }


def cross_validate_model(
    df: pd.DataFrame,
    target: str,
    *,
    model: str = "auto",
    task: Task = "auto",
    cv: int = 5,
    scale_method: str = "zscore",
    random_state: int = 0,
    **model_params: Any,
) -> dict[str, Any]:
    """Validacion cruzada con preprocesamiento dentro de cada particion."""
    from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
    from sklearn.pipeline import Pipeline

    y = df[target]
    X = df.drop(columns=[target])
    tarea = detect_task(y) if task == "auto" else task
    preprocesador = build_preprocessor(X, scale_method=scale_method)
    pipeline = Pipeline([("preprocesar", preprocesador), ("modelo", get_model(model, tarea, random_state=random_state, **model_params))])

    particiones = (
        StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        if tarea == "classification"
        else KFold(n_splits=cv, shuffle=True, random_state=random_state)
    )
    scoring = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"] if tarea == "classification" else ["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"]
    resultados = cross_validate(pipeline, X, y, cv=particiones, scoring=scoring, n_jobs=-1, return_train_score=True)

    resumen: dict[str, Any] = {"tarea": tarea, "modelo": model, "cv": cv}
    for clave, valores in resultados.items():
        if clave.startswith("test_"):
            resumen[clave.replace("test_", "")] = round(float(np.mean(valores)), 4)
            resumen[f"{clave.replace('test_', '')}_std"] = round(float(np.std(valores)), 4)
    resumen["detalle"] = {k: [round(float(v), 4) for v in vals] for k, vals in resultados.items() if k.startswith("test_")}
    return resumen


def compare_models(
    df: pd.DataFrame,
    target: str,
    *,
    models: Sequence[str] | None = None,
    task: Task = "auto",
    cv: int = 5,
    random_state: int = 0,
) -> pd.DataFrame:
    """Compara varios modelos con validacion cruzada y los ordena."""
    y = df[target]
    tarea = detect_task(y) if task == "auto" else task
    lista = list(models) if models else (["logistica", "arbol", "random_forest", "gradient_boosting"] if tarea == "classification" else ["lineal", "ridge", "arbol", "random_forest", "gradient_boosting"])
    filas: list[dict[str, Any]] = []
    for nombre in lista:
        try:
            res = cross_validate_model(df, target, model=nombre, task=tarea, cv=cv, random_state=random_state)
            fila = {"modelo": nombre, "tarea": tarea}
            fila.update({k: v for k, v in res.items() if not isinstance(v, dict) and k not in ("tarea", "modelo", "cv")})
            filas.append(fila)
        except Exception as exc:
            filas.append({"modelo": nombre, "tarea": tarea, "error": str(exc)})
    tabla = pd.DataFrame(filas)
    criterio = "accuracy" if tarea == "classification" else "r2"
    if criterio in tabla.columns:
        tabla = tabla.sort_values(criterio, ascending=False, ignore_index=True)
    return tabla


def evaluate_classification(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Metricas de clasificacion: accuracy, precision, recall, F1, ROC-AUC."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        cohen_kappa_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    resultado: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision_weighted": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "kappa": round(float(cohen_kappa_score(y_true, y_pred)), 4),
        "n": int(len(y_true)),
    }
    if y_proba is not None:
        try:
            clases = np.unique(y_true)
            if len(clases) == 2:
                resultado["roc_auc"] = round(float(roc_auc_score(y_true, y_proba[:, 1])), 4)
            else:
                resultado["roc_auc_ovr"] = round(float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")), 4)
        except Exception:
            pass
    resultado["reporte"] = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    return resultado


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    """Metricas de regresion: R2, MAE, RMSE, MAPE y error mediano."""
    from sklearn.metrics import (
        explained_variance_score,
        max_error,
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
    )

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    resultado: dict[str, Any] = {
        "r2": round(float(r2_score(y_true_arr, y_pred_arr)), 4),
        "r2_ajustado_aprox": round(float(1 - (1 - r2_score(y_true_arr, y_pred_arr)) * (len(y_true_arr) - 1) / max(len(y_true_arr) - 2, 1)), 4),
        "mae": round(float(mean_absolute_error(y_true_arr, y_pred_arr)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))), 4),
        "medae": round(float(median_absolute_error(y_true_arr, y_pred_arr)), 4),
        "varianza_explicada": round(float(explained_variance_score(y_true_arr, y_pred_arr)), 4),
        "error_maximo": round(float(max_error(y_true_arr, y_pred_arr)), 4),
        "n": int(len(y_true_arr)),
    }
    try:
        resultado["mape"] = round(float(mean_absolute_percentage_error(y_true_arr, y_pred_arr)) * 100, 4)
    except Exception:
        pass
    return resultado


def confusion_matrix_df(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """Matriz de confusion como DataFrame etiquetado."""
    from sklearn.metrics import confusion_matrix

    etiquetas = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)], ignore_index=True)).tolist(), key=str)
    matriz = confusion_matrix(y_true, y_pred, labels=etiquetas)
    return pd.DataFrame(matriz, index=[f"real_{e}" for e in etiquetas], columns=[f"pred_{e}" for e in etiquetas])


def permutation_importance(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_repeats: int = 10,
    random_state: int = 0,
    scoring: str | None = None,
) -> pd.DataFrame:
    """Importancia de cada variable original por permutacion (model-agnostic)."""
    from sklearn.inspection import permutation_importance as sk_perm

    resultado = sk_perm(pipeline, X, y, n_repeats=n_repeats, random_state=random_state, scoring=scoring, n_jobs=-1)
    tabla = pd.DataFrame(
        {
            "caracteristica": X.columns,
            "importancia_media": resultado.importances_mean,
            "desviacion": resultado.importances_std,
        }
    ).sort_values("importancia_media", ascending=False, ignore_index=True)
    return tabla.round(5)


def residuals(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """Tabla de residuos para diagnostico de regresion."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return pd.DataFrame(
        {
            "real": y_true_arr,
            "predicho": y_pred_arr,
            "residuo": y_true_arr - y_pred_arr,
            "residuo_abs": np.abs(y_true_arr - y_pred_arr),
            "residuo_pct": np.where(y_true_arr != 0, (y_true_arr - y_pred_arr) / y_true_arr * 100, np.nan),
        }
    )


def learning_curve_data(
    df: pd.DataFrame,
    target: str,
    *,
    model: str = "auto",
    task: Task = "auto",
    cv: int = 5,
    n_points: int = 6,
    random_state: int = 0,
) -> pd.DataFrame:
    """Datos de curva de aprendizaje (error de entrenamiento vs validacion)."""
    from sklearn.model_selection import KFold, StratifiedKFold, learning_curve
    from sklearn.pipeline import Pipeline

    y = df[target]
    X = df.drop(columns=[target])
    tarea = detect_task(y) if task == "auto" else task
    pipeline = Pipeline([("preprocesar", build_preprocessor(X)), ("modelo", get_model(model, tarea, random_state=random_state))])
    particiones = (
        StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        if tarea == "classification"
        else KFold(n_splits=cv, shuffle=True, random_state=random_state)
    )
    scoring = "accuracy" if tarea == "classification" else "r2"
    tamanos, train_scores, test_scores = learning_curve(
        pipeline, X, y, cv=particiones, scoring=scoring, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, n_points),
    )
    return pd.DataFrame(
        {
            "n_muestras": tamanos,
            "entrenamiento_media": train_scores.mean(axis=1),
            "entrenamiento_std": train_scores.std(axis=1),
            "validacion_media": test_scores.mean(axis=1),
            "validacion_std": test_scores.std(axis=1),
        }
    ).round(4)


# --------------------------------------------------------------------------- #
# utilidades internas
# --------------------------------------------------------------------------- #
def _proba(pipeline: Any, X: pd.DataFrame) -> np.ndarray | None:
    if hasattr(pipeline, "predict_proba"):
        try:
            return pipeline.predict_proba(X)
        except Exception:
            return None
    return None


def _feature_names(pipeline: Any, X: pd.DataFrame) -> list[str]:
    try:
        return list(pipeline.named_steps["preprocesar"].get_feature_names_out())
    except Exception:
        return list(X.columns)
