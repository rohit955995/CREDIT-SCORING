"""End-to-end credit scoring project using the UCI Statlog German Credit Data.

Run with:
    python credit_scoring_project.py

The script downloads the official UCI archive, performs EDA, trains and compares
three classifiers, tunes the strongest baseline, evaluates it, and predicts one
example customer.
"""

from pathlib import Path
import io
import zipfile
from urllib.request import urlopen

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

FEATURE_NAMES = [
    "checking_account", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment_since", "installment_rate",
    "personal_status_sex", "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing", "existing_credits",
    "job", "dependents", "telephone", "foreign_worker",
]

# UCI documents raw target 1 as good and 2 as bad. Bad risk is positive (1) here.
TARGET_NAME = "bad_risk"
CATEGORICAL_COLUMNS = [
    "checking_account", "credit_history", "purpose", "savings_account",
    "employment_since", "personal_status_sex", "other_debtors", "property",
    "other_installment_plans", "housing", "job", "telephone", "foreign_worker",
]
NUMERICAL_COLUMNS = [
    "duration_months", "credit_amount", "installment_rate", "residence_since",
    "age", "existing_credits", "dependents", "credit_amount_per_month",
    "credit_amount_per_age",
]


def download_and_load_data() -> pd.DataFrame:
    """Download the official archive and load the symbolic raw data file."""
    DATA_DIR.mkdir(exist_ok=True)
    archive_path = DATA_DIR / "statlog_german_credit_data.zip"
    if not archive_path.exists():
        print("Downloading the official UCI dataset...")
        archive_path.write_bytes(urlopen(UCI_ZIP_URL).read())

    with zipfile.ZipFile(archive_path) as archive:
        raw = archive.read("german.data")

    rows = [line.split() for line in io.BytesIO(raw).read().decode("utf-8").splitlines()]
    frame = pd.DataFrame(rows, columns=FEATURE_NAMES + ["uci_target"])
    numeric = [column for column in FEATURE_NAMES if column not in CATEGORICAL_COLUMNS]
    frame[numeric] = frame[numeric].astype(int)
    frame[TARGET_NAME] = (frame["uci_target"].astype(int) == 2).astype(int)
    return frame.drop(columns="uci_target")


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ratios that can be calculated from columns actually in this dataset."""
    result = frame.copy()
    result["credit_amount_per_month"] = (
        result["credit_amount"] / result["duration_months"].replace(0, np.nan)
    )
    result["credit_amount_per_age"] = (
        result["credit_amount"] / result["age"].replace(0, np.nan)
    )
    return result.replace([np.inf, -np.inf], np.nan)


def describe_data(frame: pd.DataFrame) -> None:
    """Print the requested data-understanding checks."""
    print("\n--- DATA UNDERSTANDING ---")
    print("First five rows:\n", frame.head())
    print("Last five rows:\n", frame.tail())
    print("Shape:", frame.shape)
    print("Columns:", list(frame.columns))
    print("Data types:\n", frame.dtypes)
    print("Summary:\n", frame.describe(include="all").T)
    print("Missing values:\n", frame.isna().sum().sort_values(ascending=False).head())
    print("Duplicate rows:", frame.duplicated().sum())
    print("Target counts (0=good, 1=bad):\n", frame[TARGET_NAME].value_counts())


def run_eda(frame: pd.DataFrame) -> None:
    """Create report-ready exploratory plots."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.countplot(data=frame, x=TARGET_NAME, ax=axes[0], hue=TARGET_NAME,
                  palette="Set2", legend=False)
    axes[0].set_title("Credit-risk class distribution")
    axes[0].set_xticks([0, 1], ["Good", "Bad"])
    sns.histplot(data=frame, x="credit_amount", hue=TARGET_NAME, bins=30,
                 element="step", stat="density", common_norm=False, ax=axes[1])
    axes[1].set_title("Credit amount by risk class")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda_overview.png", dpi=160)
    plt.close(fig)

    numeric_frame = frame.select_dtypes(include=np.number)
    correlation = numeric_frame.corr()
    fig, ax = plt.subplots(figsize=(11, 8))
    image = ax.imshow(correlation, cmap="vlag", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
    ax.set_yticks(range(len(correlation.index)), correlation.index)
    fig.colorbar(image, ax=ax, label="Correlation")
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=frame, x=TARGET_NAME, y="duration_months", hue=TARGET_NAME,
                palette="Set2", legend=False)
    plt.xticks([0, 1], ["Good", "Bad"])
    plt.title("Loan duration by credit-risk class")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "duration_by_risk.png", dpi=160)
    plt.close()


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERICAL_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )


def build_models() -> dict[str, Pipeline]:
    """Use class weights so the costlier bad-risk class is not ignored."""
    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced",
                                          random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", DecisionTreeClassifier(max_depth=5, min_samples_leaf=10,
                                              class_weight="balanced",
                                              random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                              class_weight="balanced",
                                              random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
    }


def evaluate_model(name: str, model: Pipeline, X_test: pd.DataFrame,
                   y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-Score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }


def plot_model_diagnostics(models: dict[str, Pipeline], X_test: pd.DataFrame,
                           y_test: pd.Series, results: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    best_name = results.sort_values(["Recall", "F1-Score", "ROC-AUC"],
                                    ascending=False).iloc[0]["Model"]
    best_model = models[best_name]
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test,
                                          display_labels=["Good", "Bad"], cmap="Blues",
                                          ax=axes[0])
    axes[0].set_title(f"Confusion matrix: {best_name}")

    for name, model in models.items():
        probabilities = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        auc = roc_auc_score(y_test, probabilities)
        axes[1].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", label="Random")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate",
                title="ROC curves")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_diagnostics.png", dpi=160)
    plt.close(fig)
    print("\nClassification report for selected baseline:\n",
          classification_report(y_test, best_model.predict(X_test),
                                target_names=["Good", "Bad"], zero_division=0))


def plot_feature_importance(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    importance = permutation_importance(model, X_test, y_test, scoring="f1",
                                        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    scores = pd.Series(importance.importances_mean, index=X_test.columns).sort_values()
    plt.figure(figsize=(9, 5))
    scores.tail(12).plot.barh(color="#2878b5")
    plt.title("Random Forest permutation importance (raw features)")
    plt.xlabel("Mean decrease in F1 after permutation")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=160)
    plt.close()


def main() -> None:
    frame = download_and_load_data()
    describe_data(frame)
    run_eda(frame)

    model_frame = add_features(frame)
    X = model_frame.drop(columns=TARGET_NAME)
    y = model_frame[TARGET_NAME]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    models = build_models()
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append(evaluate_model(name, model, X_test, y_test))
    results_frame = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
    print("\n--- BASELINE MODEL COMPARISON ---\n", results_frame.to_string(index=False))
    results_frame.to_csv(OUTPUT_DIR / "baseline_model_comparison.csv", index=False)
    plot_model_diagnostics(models, X_test, y_test, results_frame)
    plot_feature_importance(models["Random Forest"], X_test, y_test)

    # F1 balances missed bad customers (recall) and incorrectly flagged customers (precision).
    strongest_name = results_frame.iloc[0]["Model"]
    parameter_grids = {
        "Logistic Regression": {"model__C": [0.1, 1, 10]},
        "Decision Tree": {"model__max_depth": [3, 5, 8, None],
                           "model__min_samples_leaf": [2, 5, 10, 20]},
        "Random Forest": {"model__n_estimators": [200, 400],
                           "model__max_depth": [None, 8, 15],
                           "model__min_samples_leaf": [2, 5, 10]},
    }
    search = GridSearchCV(models[strongest_name], parameter_grids[strongest_name],
                          scoring="f1", cv=5, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    tuned = search.best_estimator_
    tuned_metrics = evaluate_model(f"Tuned {strongest_name}", tuned, X_test, y_test)
    print("\n--- TUNING ---")
    print("Strongest baseline:", strongest_name)
    print("Best parameters:", search.best_params_)
    print(pd.DataFrame([results_frame.iloc[0].to_dict(), tuned_metrics]).to_string(index=False))

    print("\n--- FINAL MODEL ---")
    print("Final model:", tuned_metrics["Model"])
    print("Selection prioritizes F1 and recall because the UCI cost matrix makes a false approval of bad risk expensive.")
    print(pd.Series(tuned_metrics).to_string())

    example = pd.DataFrame([{
        "checking_account": "A11", "duration_months": 24, "credit_history": "A32",
        "purpose": "A43", "credit_amount": 3500, "savings_account": "A61",
        "employment_since": "A73", "installment_rate": 3,
        "personal_status_sex": "A93", "other_debtors": "A101", "residence_since": 2,
        "property": "A121", "age": 35, "other_installment_plans": "A143",
        "housing": "A152", "existing_credits": 1, "job": "A173", "dependents": 1,
        "telephone": "A192", "foreign_worker": "A201",
    }])
    example_features = add_features(example)
    predicted_bad = tuned.predict(example_features)[0]
    bad_probability = tuned.predict_proba(example_features)[0, 1]
    print("\n--- NEW CUSTOMER PREDICTION ---")
    print(example.to_string(index=False))
    print("Prediction:", "Not Creditworthy (bad risk)" if predicted_bad else "Creditworthy (good risk)")
    print(f"Probability of bad risk: {bad_probability:.2%}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(tuned, OUTPUT_DIR / "final_credit_scoring_model.joblib")
    print("\nSaved charts, comparison CSV, and model to the outputs/ folder.")


if __name__ == "__main__":
    main()
