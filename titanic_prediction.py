"""
Beginner-friendly Titanic prediction script.

How to use:
  python titanic_prediction.py --input Titanic-Dataset.csv --output Titanic_Predictions.csv

This script:
- Loads the dataset
- Performs simple, explicit preprocessing (fill missing values and convert categories)
- Trains a RandomForest classifier
- Prints accuracy on a test split
- Retrains on the full dataset and saves survival probabilities for every row
"""
import argparse
import os
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------- Helper functions --------
def load_data(path):
    """Load CSV into a pandas DataFrame. Exit with a user-friendly message if missing."""
    if not os.path.exists(path):
        print(f"Error: file not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df

def preprocess(df):
    """
    Selects a few columns and converts them into numeric features.
    Steps:
      - Keep the features we care about
      - Fill missing Age/Fare with the median (simple and easy to understand)
      - Convert Sex to numbers (male -> 0, female -> 1)
      - Fill missing Embarked values and convert to dummy columns
    Returns (X, y) where X is the feature DataFrame and y is the target Series.
    """
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    # Keep only relevant columns and the target
    df = df.copy()
    missing_target = 'Survived' not in df.columns
    if missing_target:
        raise ValueError("Input CSV must contain a 'Survived' column for training/evaluation.")
    df = df[features + ['Survived']]

    # Numeric missing values -> fill with median
    for col in ['Age', 'Fare']:
        median = df[col].median()
        df[col] = df[col].fillna(median)
        print(f"Filled missing {col} with median = {median}")

    # Sex: map to 0/1
    df['Sex'] = df['Sex'].astype(str).str.lower().map({'male': 0, 'female': 1})
    df['Sex'] = df['Sex'].fillna(-1).astype(int)  # -1 means unknown

    # Embarked: fill missing and convert to dummy columns
    df['Embarked'] = df['Embarked'].fillna('Unknown').astype(str)
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df.drop(columns=['Embarked']), embarked_dummies], axis=1)

    # Final features and target
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    print("Preprocessing complete. Feature columns:", list(X.columns))
    return X, y

def train_and_evaluate(X, y, test_size=0.2, random_state=42):
    """
    Train a RandomForest on a train split, evaluate on test split, print accuracy,
    then retrain on full data for final predictions.
    Returns the model trained on the FULL dataset (use this to predict for all rows).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Split data: train={len(X_train)} rows, test={len(X_test)} rows")

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")

    # Retrain on full data to get predictions for all rows (useful for saving outputs)
    model_full = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model_full.fit(X, y)
    print("Retrained model on the full dataset for final predictions.")
    return model_full

def save_predictions(df_original, X, model, out_path):
    """Add probability and simple label to original DataFrame and save."""
    probs = model.predict_proba(X)[:, 1]  # probability of surviving
    df_out = df_original.copy()
    df_out['Survival_Probability'] = probs
    df_out['Prediction'] = df_out['Survival_Probability'].apply(
        lambda p: 'More possibility' if p > 0.5 else 'Less possibility'
    )
    df_out[['Survival_Probability', 'Prediction']].to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")
    return df_out

# -------- Main script --------
def main(args):
    df = load_data(args.input)
    X, y = preprocess(df)

    model = train_and_evaluate(X, y, test_size=args.test_size, random_state=args.random_state)

    # Save predictions for every row (using model trained on full data)
    df_with_preds = save_predictions(df, X, model, args.output)

    # Show examples
    print("\nTop 5 predicted probabilities:")
    print(df_with_preds[['Survival_Probability', 'Prediction']].head(5))
    print("\nBottom 5 predicted probabilities:")
    print(df_with_preds[['Survival_Probability', 'Prediction']].tail(5))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Titanic survival prediction (beginner-friendly).")
    parser.add_argument("--input", default="Titanic-Dataset.csv", help="Input CSV file (must include 'Survived').")
    parser.add_argument("--output", default="Titanic_Predictions.csv", help="Where to save the predictions CSV.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for the test split (0.0 - 1.0).")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    main(args)
