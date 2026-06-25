"""Shared utilities for data loading, paths, persistence, and logging."""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"
UCI_STUDENT_ZIP_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
)


def configure_logging() -> None:
    """Configure concise console logging for scripts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_directories() -> None:
    """Create project artifact directories if they do not exist."""
    for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def save_json(payload: dict, path: Path) -> None:
    """Persist a JSON artifact with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> dict:
    """Load a JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_joblib(obj: object, path: Path) -> None:
    """Save a Python artifact with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: Path) -> object:
    """Load a joblib artifact with a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def download_uci_student_dataset() -> Path:
    """Download and extract the UCI Student Performance dataset."""
    ensure_directories()
    archive_path = DATA_RAW_DIR / "student.zip"
    if not archive_path.exists():
        logging.info("Downloading UCI Student Performance dataset.")
        urlretrieve(UCI_STUDENT_ZIP_URL, archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(DATA_RAW_DIR)

    csv_path = DATA_RAW_DIR / "student-mat.csv"
    if not csv_path.exists():
        raise FileNotFoundError("Expected student-mat.csv was not found after extraction.")
    return csv_path


def standardize_uci_dataset(path: Path) -> pd.DataFrame:
    """Convert the UCI dataset schema into this project's feature schema."""
    raw = pd.read_csv(path, sep=";")
    parent_education = ((raw["Medu"] + raw["Fedu"]) / 2).round()

    df = pd.DataFrame(
        {
            "gender": raw["sex"].map({"F": "female", "M": "male"}),
            "age": raw["age"],
            "study_hours": raw["studytime"].map({1: 2, 2: 5, 3: 8, 4: 12}),
            "attendance": (100 - (raw["absences"].clip(0, 93) / 93 * 100)).round(2),
            "sleep_hours": np.select(
                [raw["health"] >= 4, raw["health"] == 3, raw["health"] <= 2],
                [8.0, 7.0, 6.0],
                default=7.0,
            ),
            "previous_grade": (((raw["G1"] + raw["G2"]) / 2) * 5).round(2),
            "internet_access": raw["internet"].map({"yes": "yes", "no": "no"}),
            "parent_education": parent_education.map(
                {0: "none", 1: "primary", 2: "middle", 3: "secondary", 4: "higher"}
            ),
            "family_income": np.select(
                [
                    (raw["address"] == "U") & (raw["famsize"] == "LE3"),
                    raw["address"] == "U",
                    raw["address"] == "R",
                ],
                ["high", "medium", "low"],
                default="medium",
            ),
            "extra_classes": np.where(
                (raw["paid"] == "yes") | (raw["schoolsup"] == "yes"), "yes", "no"
            ),
            "assignments_completed": (
                100
                - raw["failures"].clip(0, 4) * 18
                - raw["absences"].clip(0, 30) * 1.2
            ).clip(0, 100).round(2),
            "participation": (
                raw[["famrel", "freetime", "goout"]].mean(axis=1) * 20
            ).clip(0, 100).round(2),
            "final_score": (raw["G3"] * 5).clip(0, 100),
        }
    )
    df["pass_fail"] = np.where(df["final_score"] >= 40, 1, 0)
    return df


def generate_synthetic_student_data(rows: int = 600, random_state: int = 42) -> pd.DataFrame:
    """Generate a realistic fallback dataset when external data is unavailable."""
    rng = np.random.default_rng(random_state)
    study_hours = rng.normal(5.5, 2.4, rows).clip(0, 14)
    attendance = rng.normal(82, 12, rows).clip(35, 100)
    sleep_hours = rng.normal(7.1, 1.2, rows).clip(3, 11)
    previous_grade = rng.normal(68, 15, rows).clip(5, 100)
    assignments = rng.normal(78, 18, rows).clip(0, 100)
    participation = rng.normal(70, 17, rows).clip(0, 100)

    score = (
        0.32 * previous_grade
        + 2.2 * study_hours
        + 0.12 * attendance
        + 0.10 * assignments
        + 0.05 * participation
        + 1.2 * (sleep_hours - 6.5)
        - 10
        + rng.normal(0, 10, rows)
    ).clip(0, 100)

    df = pd.DataFrame(
        {
            "gender": rng.choice(["female", "male"], rows),
            "age": rng.integers(15, 20, rows),
            "study_hours": study_hours.round(2),
            "attendance": attendance.round(2),
            "sleep_hours": sleep_hours.round(2),
            "previous_grade": previous_grade.round(2),
            "internet_access": rng.choice(["yes", "no"], rows, p=[0.85, 0.15]),
            "parent_education": rng.choice(
                ["primary", "middle", "secondary", "higher"], rows, p=[0.18, 0.27, 0.35, 0.20]
            ),
            "family_income": rng.choice(["low", "medium", "high"], rows, p=[0.32, 0.48, 0.20]),
            "extra_classes": rng.choice(["yes", "no"], rows, p=[0.38, 0.62]),
            "assignments_completed": assignments.round(2),
            "participation": participation.round(2),
            "final_score": score.round(2),
        }
    )
    df["pass_fail"] = np.where(df["final_score"] >= 40, 1, 0)
    return df


def load_or_create_dataset(prefer_processed: bool = True) -> pd.DataFrame:
    """Load local processed data, UCI data, or a deterministic synthetic fallback."""
    ensure_directories()
    processed_path = DATA_PROCESSED_DIR / "student_performance_processed.csv"
    raw_standard_path = DATA_RAW_DIR / "student_performance.csv"

    if prefer_processed and processed_path.exists():
        return pd.read_csv(processed_path)
    if prefer_processed and raw_standard_path.exists():
        return pd.read_csv(raw_standard_path)

    try:
        uci_path = download_uci_student_dataset()
        data = standardize_uci_dataset(uci_path)
        data.to_csv(raw_standard_path, index=False)
        return data
    except Exception as exc:
        logging.warning("Using synthetic fallback dataset because UCI load failed: %s", exc)
        data = generate_synthetic_student_data()
        data.to_csv(raw_standard_path, index=False)
        return data
