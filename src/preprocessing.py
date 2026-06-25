"""Preprocessing utilities for model training and inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_COLUMNS = [
    "age",
    "study_hours",
    "attendance",
    "sleep_hours",
    "previous_grade",
    "assignments_completed",
    "participation",
    "study_efficiency",
    "homework_ratio",
    "academic_engagement_score",
    "sleep_quality_index",
    "grade_trend",
    "risk_index",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "internet_access",
    "parent_education",
    "family_income",
    "extra_classes",
]


@dataclass(frozen=True)
class DatasetSplit:
    """Container for regression and classification train/test splits."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_reg_train: pd.Series
    y_reg_test: pd.Series
    y_clf_train: pd.Series
    y_clf_test: pd.Series


def clean_student_data(data: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, standardize values, and fix invalid ranges."""
    df = data.copy()
    df = df.drop_duplicates().reset_index(drop=True)
    df.columns = [column.strip().lower().replace(" ", "_") for column in df.columns]

    numeric_bounds = {
        "age": (10, 25),
        "study_hours": (0, 16),
        "attendance": (0, 100),
        "sleep_hours": (0, 14),
        "previous_grade": (0, 100),
        "assignments_completed": (0, 100),
        "participation": (0, 100),
        "final_score": (0, 100),
    }
    for column, (lower, upper) in numeric_bounds.items():
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").clip(lower, upper)

    categorical_defaults = {
        "gender": "unknown",
        "internet_access": "unknown",
        "parent_education": "unknown",
        "family_income": "unknown",
        "extra_classes": "no",
    }
    for column, default in categorical_defaults.items():
        if column not in df.columns:
            df[column] = default
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace({"": default, "nan": default, "none": default})
        )

    if "pass_fail" not in df.columns and "final_score" in df.columns:
        df["pass_fail"] = np.where(df["final_score"] >= 40, 1, 0)

    return df


def build_preprocessor(
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
) -> ColumnTransformer:
    """Build the sklearn preprocessing graph.

    Numerical features are median-imputed and standardized. Categorical
    features are mode-imputed and one-hot encoded with unknown-category support
    for robust production inference.
    """
    numeric_features = numeric_columns or NUMERIC_COLUMNS
    categorical_features = categorical_columns or CATEGORICAL_COLUMNS

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_dataset(
    data: pd.DataFrame,
    feature_columns: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    """Create aligned train/test splits for regression and classification."""
    missing_targets = {"final_score", "pass_fail"} - set(data.columns)
    if missing_targets:
        raise ValueError(f"Missing required target columns: {sorted(missing_targets)}")

    x = data[feature_columns]
    y_reg = data["final_score"]
    y_clf = data["pass_fail"].astype(int)

    return DatasetSplit(
        *train_test_split(
            x,
            y_reg,
            y_clf,
            test_size=test_size,
            random_state=random_state,
            stratify=y_clf if y_clf.nunique() > 1 else None,
        )
    )
