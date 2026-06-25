"""Feature engineering for the Student Performance Predictor project."""

from __future__ import annotations

import numpy as np
import pandas as pd


ENGINEERED_FEATURES = [
    "study_efficiency",
    "homework_ratio",
    "academic_engagement_score",
    "sleep_quality_index",
    "grade_trend",
    "risk_index",
]


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create domain-driven features from cleaned student records.

    The function is intentionally pure: it returns a copy and does not mutate
    the input DataFrame. This keeps notebooks, training, and inference behavior
    consistent and testable.
    """
    df = data.copy()

    required_defaults = {
        "study_hours": 0.0,
        "attendance": 0.0,
        "assignments_completed": 0.0,
        "participation": 0.0,
        "sleep_hours": 7.0,
        "previous_grade": 0.0,
        "extra_classes": 0,
    }
    for column, default in required_defaults.items():
        if column not in df.columns:
            df[column] = default

    df["study_efficiency"] = df["study_hours"] * (df["attendance"] / 100.0)
    df["homework_ratio"] = df["assignments_completed"] / 100.0
    df["academic_engagement_score"] = (
        0.40 * (df["attendance"] / 100.0)
        + 0.30 * (df["assignments_completed"] / 100.0)
        + 0.20 * (df["participation"] / 100.0)
        + 0.10 * np.minimum(df["study_hours"] / 10.0, 1.0)
    )
    df["sleep_quality_index"] = np.clip(1.0 - (np.abs(df["sleep_hours"] - 8.0) / 8.0), 0.0, 1.0)
    df["grade_trend"] = df["previous_grade"] - df["previous_grade"].median()
    df["risk_index"] = (
        (100.0 - df["attendance"]) * 0.35
        + (100.0 - df["assignments_completed"]) * 0.25
        + np.maximum(0.0, 6.0 - df["study_hours"]) * 4.0
        + np.maximum(0.0, 6.0 - df["sleep_hours"]) * 3.0
    )

    return df


def get_model_feature_columns(data: pd.DataFrame) -> list[str]:
    """Return feature columns, excluding supervised learning targets."""
    targets = {"final_score", "pass_fail"}
    return [column for column in data.columns if column not in targets]
