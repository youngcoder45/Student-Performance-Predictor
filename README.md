<div align="center">

# Student Performance Predictor

Production-style machine learning project for predicting student final exam scores and pass/fail outcomes.

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

## Project Overview

Student Performance Predictor is an end-to-end machine learning application that solves two related academic analytics problems:

- Regression: predict a student's final exam score from academic, behavioral, and demographic inputs.
- Classification: predict whether the student is likely to pass or fail.

The project is structured as a portfolio-ready ML system, not a single notebook. It includes data ingestion, cleaning, EDA, feature engineering, model comparison, hyperparameter tuning, model persistence, and an interactive Streamlit application.

## Features

- UCI Student Performance dataset integration with a deterministic offline fallback dataset.
- Data quality handling for missing values, duplicates, invalid values, categorical encoding, and scaling.
- Engineered academic features such as study efficiency, homework ratio, engagement score, sleep quality index, grade trend, and risk index.
- Regression models: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and optional XGBoost.
- Classification models: Logistic Regression, Decision Tree, Random Forest, SVM, Gradient Boosting, and optional XGBoost.
- Hyperparameter tuning with `RandomizedSearchCV`.
- Saved `joblib` artifacts for regression, classification, preprocessing, encoder, and scaler.
- Streamlit interface with score prediction, pass/fail probability, confidence score, and feature contribution chart.
- Reproducible notebooks for EDA and model training.

## Screenshots

Add screenshots after running the Streamlit app:

- `screenshots/app_home.png`
- `screenshots/prediction_result.png`
- `reports/figures/regression_feature_importance.png`
- `reports/figures/classification_feature_importance.png`
- `reports/figures/confusion_matrix.png`
- `reports/figures/roc_curve.png`

## Dataset

Primary source: UCI Machine Learning Repository, Student Performance Datasets.

The training code attempts to download:

```text
https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip
```

The UCI data contains Portuguese school performance records. This project standardizes the source fields into a clean portfolio schema:

- `gender`
- `age`
- `study_hours`
- `attendance`
- `sleep_hours`
- `previous_grade`
- `internet_access`
- `parent_education`
- `family_income`
- `extra_classes`
- `assignments_completed`
- `participation`
- `final_score`
- `pass_fail`

If the dataset cannot be downloaded, `src/utils.py` creates a deterministic synthetic dataset so the project still trains and runs offline.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Project Structure

```text
student-performance-predictor/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── EDA.ipynb
│   └── Model_Training.ipynb
├── models/
│   ├── regression.pkl
│   ├── classifier.pkl
│   ├── preprocessor.pkl
│   ├── encoders_scaler.pkl
│   └── metrics.json
├── reports/
│   └── figures/
├── screenshots/
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── app.py
├── requirements.txt
├── pyproject.toml
├── README.md
├── LICENSE
└── .gitignore
```

## Machine Learning Workflow

1. Load UCI data or create the offline fallback dataset.
2. Standardize source columns into a consistent application schema.
3. Remove duplicates and correct invalid numeric ranges.
4. Create engineered features:
   - Study Efficiency = Study Hours x Attendance
   - Homework Ratio
   - Academic Engagement Score
   - Sleep Quality Index
   - Grade Trend
   - Risk Index
5. Split data into train and test sets.
6. Build sklearn pipelines with median imputation, categorical imputation, standard scaling, and one-hot encoding.
7. Train multiple regression and classification models.
8. Tune final Random Forest models with `RandomizedSearchCV`.
9. Evaluate with regression and classification metrics.
10. Save models, preprocessing artifacts, metrics, and diagnostic plots.
11. Serve predictions through Streamlit.

## Exploratory Data Analysis

The EDA notebook includes:

- Missing-value audit
- Duplicate-record check
- Summary statistics
- IQR outlier detection
- Histograms
- Box plots
- Final-score distribution
- Pass/fail distribution
- Correlation heatmap
- Pair plot
- Categorical distributions
- Feature importance preview

Each graph is followed by an interpretation explaining what the visualization contributes to model design.

## Algorithms Used

Regression:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor, when installed

Classification:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine
- Gradient Boosting Classifier
- XGBoost Classifier, when installed

## Evaluation Metrics

Regression:

- MAE
- MSE
- RMSE
- R2 Score

Classification:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- Confusion Matrix

## Run Training

```bash
python -m src.train
```

Generated artifacts:

- `models/regression.pkl`
- `models/classifier.pkl`
- `models/preprocessor.pkl`
- `models/encoders_scaler.pkl`
- `models/metrics.json`
- `reports/figures/regression_model_comparison.csv`
- `reports/figures/classification_model_comparison.csv`
- `reports/figures/regression_feature_importance.png`
- `reports/figures/classification_feature_importance.png`
- `reports/figures/confusion_matrix.png`
- `reports/figures/roc_curve.png`

## Run the Web App

```bash
streamlit run app.py
```

The app sidebar accepts student details and returns:

- Predicted final score
- Pass/fail result
- Pass probability
- Confidence score
- Feature contribution chart

## Results

The latest training run used the UCI Student Performance dataset and selected the best evaluated candidate from the baseline and tuned model set.

Selected regression model: Random Forest Regressor

| Metric | Value |
| --- | ---: |
| MAE | 5.4359 |
| MSE | 61.7454 |
| RMSE | 7.8578 |
| R2 | 0.8801 |

Selected classification model: Gradient Boosting Classifier

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9494 |
| Precision | 0.9552 |
| Recall | 0.9846 |
| F1 Score | 0.9697 |

The final metrics are saved to `models/metrics.json`, and comparison tables are saved under `reports/figures/`.

The expected strong predictors are previous grades, attendance, study efficiency, assignments completed, engagement score, and risk index.

## Future Improvements

- Add model monitoring for prediction drift.
- Add SHAP explanations for richer local interpretability.
- Add a REST API with FastAPI.
- Add automated CI checks for formatting and training smoke tests.
- Expand the dataset with school-specific and semester-specific records.
- Add threshold optimization for pass/fail decisions.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Author

Aditya Verma

Portfolio project for GitHub, LinkedIn, and internship applications.
