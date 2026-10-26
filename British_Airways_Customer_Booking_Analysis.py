"""British Airways: Predicting Customer Buying Behaviour

Run:
    pip install pandas numpy matplotlib scikit-learn
    python British_Airways_Customer_Booking_Analysis.py

Keep customer_booking.csv in the same folder as this script.
This analysis is for the Forage virtual job simulation.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    classification_report,
)
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
CSV_PATH = Path("customer_booking.csv")


def main():
    # 1. Load and inspect dataset
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            "customer_booking.csv was not found. Put it in the same folder as this script."
        )

    df = pd.read_csv(CSV_PATH, encoding="ISO-8859-1")
    print("DATA OVERVIEW")
    print("-------------")
    print("Shape:", df.shape)
    print("Missing values:", int(df.isna().sum().sum()))
    print("Booking outcome counts:\n", df["booking_complete"].value_counts().sort_index())
    print(f"Completed booking rate: {df['booking_complete'].mean():.2%}\n")

    # 2. Feature engineering
    day_map = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}
    df["flight_day_num"] = df["flight_day"].map(day_map)
    df["purchase_lead_band"] = pd.cut(
        df["purchase_lead"], bins=[-1, 7, 30, 90, np.inf],
        labels=["0-7d", "8-30d", "31-90d", ">90d"]
    )
    df["length_of_stay_band"] = pd.cut(
        df["length_of_stay"], bins=[-1, 3, 7, 14, np.inf],
        labels=["0-3d", "4-7d", "8-14d", ">14d"]
    )
    df["departure_period"] = pd.cut(
        df["flight_hour"], bins=[-1, 5, 11, 17, 23],
        labels=["Overnight/early", "Morning", "Afternoon", "Evening"],
        include_lowest=True,
    )

    target = "booking_complete"
    X = df.drop(columns=[target, "flight_day"])
    y = df[target].astype(int)
    cat_cols = [
        "sales_channel", "trip_type", "route", "booking_origin",
        "purchase_lead_band", "length_of_stay_band", "departure_period"
    ]
    num_cols = [col for col in X.columns if col not in cat_cols]

    # Split first. Preprocessing is inside the pipeline to avoid leakage from test folds.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=22,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])

    # 3. Five-fold stratified cross-validation on the training split
    print("CROSS-VALIDATION (5 folds)")
    print("--------------------------")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "avg_precision": "average_precision",
    }
    cv_results = cross_validate(
        pipe, X_train, y_train, cv=cv, scoring=scoring,
        n_jobs=1, return_train_score=False
    )
    for key in scoring:
        vals = cv_results[f"test_{key}"]
        print(f"{key:18s}: {vals.mean():.3f} +/- {vals.std():.3f}")

    # 4. Fit final model on training set and evaluate untouched holdout data
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    threshold = 0.50
    y_pred = (y_prob >= threshold).astype(int)
    holdout = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Average precision": average_precision_score(y_test, y_prob),
    }
    print("\nHOLDOUT TEST METRICS")
    print("--------------------")
    for key, value in holdout.items():
        print(f"{key:18s}: {value:.3f}")
    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    # 5. Permutation feature importance on a fixed holdout sample
    print("PERMUTATION FEATURE IMPORTANCE")
    print("-------------------------------")
    n_perm = min(3000, len(X_test))
    X_perm = X_test.sample(n=n_perm, random_state=RANDOM_STATE)
    y_perm = y_test.loc[X_perm.index]
    perm = permutation_importance(
        pipe, X_perm, y_perm, n_repeats=3, random_state=RANDOM_STATE,
        scoring="average_precision", n_jobs=1
    )
    importance = pd.DataFrame({
        "Feature": X_perm.columns,
        "Mean drop in average precision": perm.importances_mean,
        "Std. dev.": perm.importances_std,
    }).sort_values("Mean drop in average precision", ascending=False)
    print(importance.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    importance.to_csv("booking_feature_importance.csv", index=False)

    top = importance.head(8).sort_values("Mean drop in average precision")
    plt.figure(figsize=(8, 5))
    plt.barh(top["Feature"], top["Mean drop in average precision"], color="#C8102E")
    plt.xlabel("Decrease in average precision when feature is shuffled")
    plt.title("Feature permutation importance", loc="left", weight="bold")
    plt.grid(axis="x", alpha=0.2)
    plt.gca().set_axisbelow(True)
    plt.tight_layout()
    plt.savefig("booking_feature_importance.png", dpi=180, bbox_inches="tight")
    print("\nSaved: booking_feature_importance.csv and booking_feature_importance.png")
    plt.show()


if __name__ == "__main__":
    main()
