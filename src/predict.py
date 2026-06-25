"""Prediction utilities used by the Streamlit app and external callers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.feature_engineering import add_engineered_features
from src.preprocessing import clean_student_data
from src.utils import MODELS_DIR, load_joblib


@dataclass(frozen=True)
class PredictionResult:
    """Structured prediction response for one student."""

    predicted_score: float
    pass_fail: str
    pass_probability: float
    confidence_score: float
    contributions: pd.DataFrame


def load_models() -> tuple[Pipeline, Pipeline]:
    """Load trained regression and classification pipelines."""
    regression_model = load_joblib(MODELS_DIR / "regression.pkl")
    classifier_model = load_joblib(MODELS_DIR / "classifier.pkl")
    return regression_model, classifier_model


def build_student_frame(
    age: int,
    gender: str,
    attendance: float,
    study_hours: float,
    sleep_hours: float,
    assignments_completed: float,
    previous_grade: float,
    parent_education: str,
    internet_access: str = "yes",
    family_income: str = "medium",
    extra_classes: str = "no",
    participation: float = 70.0,
) -> pd.DataFrame:
    """Create a single-row DataFrame from user input."""
    return pd.DataFrame(
        [
            {
                "age": age,
                "gender": gender,
                "attendance": attendance,
                "study_hours": study_hours,
                "sleep_hours": sleep_hours,
                "assignments_completed": assignments_completed,
                "previous_grade": previous_grade,
                "parent_education": parent_education,
                "internet_access": internet_access,
                "family_income": family_income,
                "extra_classes": extra_classes,
                "participation": participation,
            }
        ]
    )


def prepare_inference_frame(student_data: pd.DataFrame) -> pd.DataFrame:
    """Apply the same cleaning and feature engineering used in training."""
    cleaned = clean_student_data(student_data)
    return add_engineered_features(cleaned)


def estimate_feature_contributions(model: Pipeline, data: pd.DataFrame) -> pd.DataFrame:
    """Estimate top local feature contributions from transformed input values."""
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    transformed = preprocessor.transform(data)
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(estimator, "feature_importances_"):
        weights = estimator.feature_importances_
        signed_values = transformed[0] * weights
    elif hasattr(estimator, "coef_"):
        weights = np.ravel(estimator.coef_)
        signed_values = transformed[0] * weights
    else:
        return pd.DataFrame(columns=["feature", "contribution"])

    contributions = (
        pd.DataFrame({"feature": feature_names, "contribution": signed_values})
        .assign(abs_contribution=lambda df: df["contribution"].abs())
        .sort_values("abs_contribution", ascending=False)
        .head(10)
        .drop(columns="abs_contribution")
        .sort_values("contribution")
    )
    return contributions


def predict_student(student_data: pd.DataFrame) -> PredictionResult:
    """Predict final score, pass/fail label, probability, and contributions."""
    regression_model, classifier_model = load_models()
    prepared = prepare_inference_frame(student_data)

    predicted_score = float(np.clip(regression_model.predict(prepared)[0], 0, 100))
    class_prediction = int(classifier_model.predict(prepared)[0])

    if hasattr(classifier_model, "predict_proba"):
        pass_probability = float(classifier_model.predict_proba(prepared)[0][1])
    else:
        pass_probability = float(class_prediction)

    confidence_score = pass_probability if class_prediction == 1 else 1.0 - pass_probability
    return PredictionResult(
        predicted_score=round(predicted_score, 2),
        pass_fail="Pass" if class_prediction == 1 else "Fail",
        pass_probability=round(pass_probability, 4),
        confidence_score=round(confidence_score, 4),
        contributions=estimate_feature_contributions(regression_model, prepared),
    )
