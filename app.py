"""Streamlit application for interactive student performance prediction."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.predict import build_student_frame, predict_student
from src.utils import MODELS_DIR


st.set_page_config(
    page_title="Student Performance Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)


def reset_inputs() -> None:
    """Reset Streamlit widget state to sensible defaults."""
    defaults = {
        "age": 17,
        "gender": "female",
        "attendance": 85,
        "study_hours": 5.0,
        "sleep_hours": 7.5,
        "assignments_completed": 80,
        "previous_grade": 70,
        "parent_education": "secondary",
        "internet_access": "yes",
        "family_income": "medium",
        "extra_classes": "no",
        "participation": 70,
    }
    for key, value in defaults.items():
        st.session_state[key] = value


if "age" not in st.session_state:
    reset_inputs()

st.title("Student Performance Predictor")
st.caption("Regression score prediction and pass/fail classification from academic inputs.")

if not (MODELS_DIR / "regression.pkl").exists() or not (MODELS_DIR / "classifier.pkl").exists():
    st.warning("Model artifacts are missing. Run `python -m src.train` before using the app.")
    st.stop()

with st.sidebar:
    st.header("Student Details")
    age = st.slider("Age", min_value=10, max_value=25, key="age")
    gender = st.selectbox("Gender", ["female", "male", "unknown"], key="gender")
    attendance = st.slider("Attendance (%)", min_value=0, max_value=100, key="attendance")
    study_hours = st.number_input(
        "Study Hours / Day",
        min_value=0.0,
        max_value=16.0,
        step=0.5,
        key="study_hours",
    )
    sleep_hours = st.number_input(
        "Sleep Hours / Day",
        min_value=0.0,
        max_value=14.0,
        step=0.5,
        key="sleep_hours",
    )
    assignments_completed = st.slider(
        "Assignments Completed (%)",
        min_value=0,
        max_value=100,
        key="assignments_completed",
    )
    previous_grade = st.slider("Previous Grade (%)", min_value=0, max_value=100, key="previous_grade")
    parent_education = st.selectbox(
        "Parent Education",
        ["none", "primary", "middle", "secondary", "higher", "unknown"],
        key="parent_education",
    )
    internet_access = st.selectbox("Internet Access", ["yes", "no", "unknown"], key="internet_access")
    family_income = st.selectbox("Family Income", ["low", "medium", "high", "unknown"], key="family_income")
    extra_classes = st.selectbox("Extra Classes", ["no", "yes"], key="extra_classes")
    participation = st.slider("Participation (%)", min_value=0, max_value=100, key="participation")

    predict_score_clicked = st.button("Predict Score", use_container_width=True)
    predict_class_clicked = st.button("Predict Pass/Fail", use_container_width=True)
    if st.button("Reset", use_container_width=True):
        reset_inputs()
        st.rerun()

student = build_student_frame(
    age=age,
    gender=gender,
    attendance=attendance,
    study_hours=study_hours,
    sleep_hours=sleep_hours,
    assignments_completed=assignments_completed,
    previous_grade=previous_grade,
    parent_education=parent_education,
    internet_access=internet_access,
    family_income=family_income,
    extra_classes=extra_classes,
    participation=participation,
)

if predict_score_clicked or predict_class_clicked:
    result = predict_student(student)

    score_col, status_col, probability_col, confidence_col = st.columns(4)
    score_col.metric("Predicted Score", f"{result.predicted_score:.2f}%")
    status_col.metric("Pass/Fail Result", result.pass_fail)
    probability_col.metric("Prediction Probability", f"{result.pass_probability * 100:.1f}%")
    confidence_col.metric("Confidence Score", f"{result.confidence_score * 100:.1f}%")

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Feature Contributions")
        if result.contributions.empty:
            st.info("Feature contribution chart is unavailable for this estimator.")
        else:
            figure = px.bar(
                result.contributions,
                x="contribution",
                y="feature",
                orientation="h",
                color="contribution",
                color_continuous_scale="RdBu",
                title="Top Local Contributions to Predicted Score",
            )
            figure.update_layout(height=480, yaxis_title="", xaxis_title="Contribution")
            st.plotly_chart(figure, use_container_width=True)

    with right:
        st.subheader("Student Profile")
        profile = student.T.rename(columns={0: "value"})
        st.dataframe(profile, use_container_width=True)
else:
    st.info("Enter student details in the sidebar and run a prediction.")
