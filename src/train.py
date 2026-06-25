"""Train, evaluate, tune, and save student performance models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover - optional dependency guard
    XGBClassifier = None
    XGBRegressor = None

from src.feature_engineering import add_engineered_features, get_model_feature_columns
from src.preprocessing import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    build_preprocessor,
    clean_student_data,
    split_dataset,
)
from src.utils import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    configure_logging,
    ensure_directories,
    load_or_create_dataset,
    save_joblib,
    save_json,
)


RANDOM_STATE = 42


def regression_models() -> dict[str, BaseEstimator]:
    """Return baseline regression estimators."""
    models: dict[str, BaseEstimator] = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=250,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    if XGBRegressor is not None:
        models["XGBoost Regressor"] = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
        )
    return models


def classification_models() -> dict[str, BaseEstimator]:
    """Return baseline classification estimators."""
    models: dict[str, BaseEstimator] = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=250,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "SVM": CalibratedClassifierCV(
            estimator=SVC(random_state=RANDOM_STATE),
            method="sigmoid",
            cv=3,
        ),
        "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    if XGBClassifier is not None:
        models["XGBoost Classifier"] = XGBClassifier(
            eval_metric="logloss",
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
        )
    return models


def make_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Create a model pipeline with preprocessing and estimator."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(NUMERIC_COLUMNS, CATEGORICAL_COLUMNS)),
            ("model", estimator),
        ]
    )


def evaluate_regression(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Calculate regression metrics."""
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    return {
        "MAE": round(mean_absolute_error(y_test, predictions), 4),
        "MSE": round(mse, 4),
        "RMSE": round(float(np.sqrt(mse)), 4),
        "R2": round(r2_score(y_test, predictions), 4),
    }


def evaluate_classification(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Calculate classification metrics."""
    predictions = model.predict(x_test)
    return {
        "Accuracy": round(accuracy_score(y_test, predictions), 4),
        "Precision": round(precision_score(y_test, predictions, zero_division=0), 4),
        "Recall": round(recall_score(y_test, predictions, zero_division=0), 4),
        "F1": round(f1_score(y_test, predictions, zero_division=0), 4),
    }


def train_baselines(
    models: dict[str, BaseEstimator],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    task: str,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Train model families and return comparison metrics."""
    fitted_models: dict[str, Pipeline] = {}
    rows: list[dict[str, Any]] = []

    for name, estimator in models.items():
        logging.info("Training %s", name)
        pipeline = make_pipeline(estimator)
        pipeline.fit(x_train, y_train)
        fitted_models[name] = pipeline

        metrics = (
            evaluate_regression(pipeline, x_test, y_test)
            if task == "regression"
            else evaluate_classification(pipeline, x_test, y_test)
        )
        rows.append({"Model": name, **metrics})

    return pd.DataFrame(rows), fitted_models


def tune_regression_model(x_train: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
    """Tune a Random Forest regressor with a compact but meaningful search."""
    search = RandomizedSearchCV(
        estimator=make_pipeline(RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
        param_distributions={
            "model__n_estimators": [150, 250, 400, 600],
            "model__max_depth": [None, 4, 6, 10, 14],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.8],
        },
        n_iter=20,
        scoring="neg_root_mean_squared_error",
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    return search


def tune_classifier_model(x_train: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
    """Tune a Random Forest classifier with class-aware scoring."""
    search = RandomizedSearchCV(
        estimator=make_pipeline(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        param_distributions={
            "model__n_estimators": [150, 250, 400, 600],
            "model__max_depth": [None, 4, 6, 10, 14],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.8],
            "model__class_weight": [None, "balanced"],
        },
        n_iter=20,
        scoring="f1",
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    return search


def save_feature_importance(model: Pipeline, output_path: Path, title: str) -> None:
    """Save a feature importance chart for tree models or linear coefficients."""
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.ravel(np.abs(estimator.coef_))
    else:
        return

    importance = (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .head(15)
        .sort_values("importance")
    )

    plt.figure(figsize=(10, 7))
    plt.barh(importance["feature"], importance["importance"], color="#1f77b4")
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_classification_diagnostics(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Save confusion matrix and ROC curve plots."""
    ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, cmap="Blues")
    plt.title("Classifier Confusion Matrix")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    if hasattr(model, "predict_proba"):
        RocCurveDisplay.from_estimator(model, x_test, y_test)
        plt.title("Classifier ROC Curve")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "roc_curve.png", dpi=180)
        plt.close()


def prepare_dataset() -> pd.DataFrame:
    """Load, clean, engineer, and persist the modeling dataset."""
    raw_data = load_or_create_dataset(prefer_processed=False)
    cleaned = clean_student_data(raw_data)
    engineered = add_engineered_features(cleaned)
    output_path = DATA_PROCESSED_DIR / "student_performance_processed.csv"
    engineered.to_csv(output_path, index=False)
    return engineered


def main() -> None:
    """Run the complete training workflow."""
    configure_logging()
    ensure_directories()

    data = prepare_dataset()
    feature_columns = get_model_feature_columns(data)
    split = split_dataset(data, feature_columns)

    regression_table, regression_fitted = train_baselines(
        regression_models(),
        split.x_train,
        split.y_reg_train,
        split.x_test,
        split.y_reg_test,
        task="regression",
    )
    classification_table, classification_fitted = train_baselines(
        classification_models(),
        split.x_train,
        split.y_clf_train,
        split.x_test,
        split.y_clf_test,
        task="classification",
    )

    logging.info("Tuning final regression model.")
    tuned_regression = tune_regression_model(split.x_train, split.y_reg_train)
    tuned_regression_model = tuned_regression.best_estimator_
    tuned_regression_metrics = evaluate_regression(
        tuned_regression_model,
        split.x_test,
        split.y_reg_test,
    )

    logging.info("Tuning final classification model.")
    tuned_classifier = tune_classifier_model(split.x_train, split.y_clf_train)
    tuned_classifier_model = tuned_classifier.best_estimator_
    tuned_classifier_metrics = evaluate_classification(
        tuned_classifier_model,
        split.x_test,
        split.y_clf_test,
    )

    regression_table = pd.concat(
        [
            regression_table,
            pd.DataFrame([{"Model": "Tuned Random Forest Regressor", **tuned_regression_metrics}]),
        ],
        ignore_index=True,
    )
    classification_table = pd.concat(
        [
            classification_table,
            pd.DataFrame([{"Model": "Tuned Random Forest Classifier", **tuned_classifier_metrics}]),
        ],
        ignore_index=True,
    )

    selected_regression_row = regression_table.sort_values("RMSE", ascending=True).iloc[0]
    selected_classifier_row = classification_table.sort_values("F1", ascending=False).iloc[0]
    selected_regression_name = str(selected_regression_row["Model"])
    selected_classifier_name = str(selected_classifier_row["Model"])

    regression_candidates = {
        **regression_fitted,
        "Tuned Random Forest Regressor": tuned_regression_model,
    }
    classifier_candidates = {
        **classification_fitted,
        "Tuned Random Forest Classifier": tuned_classifier_model,
    }
    regression_model = regression_candidates[selected_regression_name]
    classifier_model = classifier_candidates[selected_classifier_name]
    selected_regression_metrics = evaluate_regression(
        regression_model,
        split.x_test,
        split.y_reg_test,
    )
    selected_classifier_metrics = evaluate_classification(
        classifier_model,
        split.x_test,
        split.y_clf_test,
    )

    regression_table.to_csv(REPORTS_DIR / "regression_model_comparison.csv", index=False)
    classification_table.to_csv(REPORTS_DIR / "classification_model_comparison.csv", index=False)

    save_joblib(regression_model, MODELS_DIR / "regression.pkl")
    save_joblib(classifier_model, MODELS_DIR / "classifier.pkl")
    save_joblib(regression_model.named_steps["preprocessor"], MODELS_DIR / "preprocessor.pkl")
    save_joblib(
        {
            "numeric_scaler": regression_model.named_steps["preprocessor"]
            .named_transformers_["num"]
            .named_steps["scaler"],
            "categorical_encoder": regression_model.named_steps["preprocessor"]
            .named_transformers_["cat"]
            .named_steps["encoder"],
        },
        MODELS_DIR / "encoders_scaler.pkl",
    )

    save_feature_importance(
        regression_model,
        REPORTS_DIR / "regression_feature_importance.png",
        "Regression Feature Importance",
    )
    save_feature_importance(
        classifier_model,
        REPORTS_DIR / "classification_feature_importance.png",
        "Classification Feature Importance",
    )
    save_classification_diagnostics(classifier_model, split.x_test, split.y_clf_test)

    save_json(
        {
            "regression_best_params": tuned_regression.best_params_,
            "classification_best_params": tuned_classifier.best_params_,
            "selected_regression_model": selected_regression_name,
            "selected_classifier_model": selected_classifier_name,
            "regression_metrics": selected_regression_metrics,
            "classification_metrics": selected_classifier_metrics,
            "feature_columns": feature_columns,
            "pass_threshold": 40,
        },
        MODELS_DIR / "metrics.json",
    )

    logging.info("Training complete. Artifacts saved in %s", MODELS_DIR)
    logging.info("Regression comparison:\n%s", regression_table.to_string(index=False))
    logging.info("Classification comparison:\n%s", classification_table.to_string(index=False))


if __name__ == "__main__":
    main()
