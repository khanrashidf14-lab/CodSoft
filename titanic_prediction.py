#!/usr/bin/env python3
"""
train_titanic.py

Usage:
    python train_titanic.py

Requires:
    - Titanic-Dataset.csv in same directory
    - Install dependencies from requirements.txt
"""

import re
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
DATA_FILE = "Titanic-Dataset.csv"
OUTPUT_MODEL = "best_model.pkl"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def extract_title(name: str) -> str:
    # Extract title between ", " and "."
    m = re.search(r",\s*([^\.]+)\.", name)
    if m:
        return m.group(1).strip()
    return "Unknown"


def preprocess(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    # Copy
    df = df.copy()

    # Target
    y = df["Survived"].astype(int)

    # Feature engineering
    df["Title"] = df["Name"].map(lambda n: extract_title(n))
    # Simplify titles
    title_map = {
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        "Lady": "Royal", "Countess": "Royal", "Don": "Mr",
        "Sir": "Mr", "Jonkheer": "Mr", "Dona": "Mrs",
        "Col": "Officer", "Major": "Officer", "Capt": "Officer",
        "Rev": "Officer", "Dr": "Officer"
    }
    df["Title"] = df["Title"].replace(title_map)
    rare_titles = df["Title"].value_counts()[df["Title"].value_counts() < 10].index
    df.loc[df["Title"].isin(rare_titles), "Title"] = "Rare"

    df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Deck from Cabin (first letter), missing -> 'U' unknown
    df["Deck"] = df["Cabin"].astype(str).str[0].replace("n", "U").replace("N", "U")
    df.loc[df["Deck"].isna(), "Deck"] = "U"
    df["Deck"] = df["Deck"].fillna("U")
    df.loc[df["Deck"] == "n", "Deck"] = "U"

    # Embarked: fill with mode
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Fare: fill missing with median (if any)
    if df["Fare"].isna().any():
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Age: fill missing using median by Title and Pclass, fallback to median
    age_medians = df.groupby(["Title", "Pclass"])["Age"].median()
    def fill_age(row):
        if pd.notna(row["Age"]):
            return row["Age"]
        key = (row["Title"], row["Pclass"])
        if key in age_medians and not np.isnan(age_medians.loc[key]):
            return age_medians.loc[key]
        return df["Age"].median()
    df["Age"] = df.apply(fill_age, axis=1)

    # Optional transforms
    df["Fare_log1p"] = np.log1p(df["Fare"])

    # Drop less useful or leaky columns
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin", "Survived"]
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    # Select features to use
    # numeric features
    numeric_feats = ["Age", "Fare", "Fare_log1p", "FamilySize", "SibSp", "Parch"]
    # categorical features
    categorical_feats = ["Pclass", "Sex", "Embarked", "Title", "Deck", "IsAlone"]

    features = numeric_feats + categorical_feats
    X = df[features].copy()

    # One-hot encode categorical features using pandas.get_dummies
    X = pd.get_dummies(X, columns=[c for c in categorical_feats], drop_first=False)

    # Fill any remaining NaNs
    X = X.fillna(X.median())

    return X, y


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # some classifiers produce predict_proba, handle gracefully for roc_auc
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc = float("nan")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy: {:.4f}  Precision: {:.4f}  Recall: {:.4f}  F1: {:.4f}  ROC AUC: {:.4f}".format(acc, prec, rec, f1, roc))
    print("Confusion matrix:\n", cm)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc, "confusion_matrix": cm}


def main():
    p = Path(DATA_FILE)
    if not p.exists():
        raise FileNotFoundError(f"{DATA_FILE} not found. Place Titanic-Dataset.csv in current directory.")

    print("Loading data...")
    df = load_data(DATA_FILE)
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

    print("Preprocessing and feature engineering...")
    X, y = preprocess(df)
    print("Feature shape:", X.shape)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # Scale numeric features (optional but helpful for logistic regression)
    numeric_cols = [c for c in X.columns if X[c].dtype in [np.float64, np.int64] and c in ["Age", "Fare", "Fare_log1p", "FamilySize", "SibSp", "Parch"]]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Baseline 1: Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)
    print("Logistic Regression evaluation on test set:")
    lr_metrics = evaluate_model(lr, X_test_scaled, y_test)

    # Baseline 2: Random Forest (no scaling required)
    print("\nTraining Random Forest (with simple grid search)...")
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    # quick grid for demonstration — increase for serious tuning
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [6, 10, None],
        "min_samples_split": [2, 5]
    }
    grid = GridSearchCV(rf, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_
    print("Best RF params:", grid.best_params_)
    print("Random Forest evaluation on test set:")
    rf_metrics = evaluate_model(best_rf, X_test, y_test)

    # Choose best by F1
    best_model = lr if lr_metrics["f1"] >= rf_metrics["f1"] else best_rf
    best_name = "LogisticRegression" if best_model is lr else "RandomForest"
    print(f"\nSelected best model: {best_name}")

    # Save model (and scaler + columns) for later inference
    model_bundle = {
        "model": best_model,
        "scaler": scaler,
        "numeric_cols": numeric_cols,
        "columns": X.columns.tolist()
    }
    with open(OUTPUT_MODEL, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"Saved best model bundle to {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
