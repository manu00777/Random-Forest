"""
Titanic Survival Prediction using Random Forest
Dataset: titanic.csv (Kaggle-style Titanic dataset)
Run: python main.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "titanic.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "survived"
FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
NUMERIC_FEATURES = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL_FEATURES = ["sex", "embarked"]


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place titanic.csv in the same folder as main.py."
        )

    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "The CSV is missing required columns: "
            + ", ".join(missing_columns)
            + "\nExpected columns include: "
            + ", ".join(required)
        )

    df = df[required].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    invalid_targets = sorted(set(df[TARGET].unique()) - {0, 1})
    if invalid_targets:
        raise ValueError(f"Target '{TARGET}' must contain only 0 and 1.")

    return df


def build_pipeline():
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[("preprocessing", preprocessing), ("classifier", model)]
    )


def save_feature_importance(pipeline):
    preprocessor = pipeline.named_steps["preprocessing"]
    classifier = pipeline.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    importance = pd.Series(
        classifier.feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", header=["importance"])

    plt.figure(figsize=(10, 6))
    importance.head(12).sort_values().plot(kind="barh")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close()


def main():
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print("\nFirst five rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    print(f"\nTest accuracy: {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Did not survive", "Survived"],
            zero_division=0,
        )
    )

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    print("Confusion matrix:")
    print(matrix)

    if y_test.nunique() == 2:
        print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        build_pipeline(), X, y, cv=cv, scoring="accuracy", n_jobs=-1
    )
    print(
        f"\n5-fold CV accuracy: {cv_scores.mean() * 100:.2f}% "
        f"(+/- {cv_scores.std() * 100:.2f}%)"
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Did not survive", "Survived"],
    )
    display.plot(cmap="Purples", colorbar=False)
    plt.title("Titanic Survival - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    save_feature_importance(pipeline)

    # Example passenger: 1st class, female, age 25, fare 80, embarked at C.
    new_passenger = pd.DataFrame(
        [{
            "pclass": 1,
            "sex": "female",
            "age": 25,
            "sibsp": 0,
            "parch": 0,
            "fare": 80.0,
            "embarked": "C",
        }]
    )

    prediction = int(pipeline.predict(new_passenger)[0])
    probability = float(pipeline.predict_proba(new_passenger)[0, 1])
    label = "Survived" if prediction == 1 else "Did not survive"

    print("\nExample passenger prediction:")
    print(f"Prediction: {label}")
    print(f"Estimated survival probability: {probability * 100:.2f}%")

    results = X_test.copy()
    results["actual_survived"] = y_test
    results["predicted_survived"] = predictions
    results["survival_probability"] = probabilities
    results.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    print(f"\nOutput files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
