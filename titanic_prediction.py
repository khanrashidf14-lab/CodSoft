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
)
