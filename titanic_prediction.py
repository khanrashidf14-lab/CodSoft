"""
Beginner-friendly Titanic prediction script (now self-contained / independent).

How to use (examples):
  # Train (if input CSV contains 'Survived' this will train; mode=train forces training):
  python titanic_prediction.py --input Titanic-Dataset.csv --output Titanic_Predictions.csv --mode train --model titanic_model.joblib

  # Predict (input CSV does NOT contain 'Survived'):
  python titanic_prediction.py --input Titanic-Unlabeled.csv --output Titanic_Predictions.csv --mode predict --model titanic_model.joblib

  # Auto mode (default): if 'Survived' in CSV -> train & save model; otherwise -> predict using existing model
  python titanic_prediction.py --input Titanic-Dataset.csv --output Titanic_Predictions.csv --model titanic_model.joblib

What changed to make the script "independent":
- Supports both training and prediction modes.
- Saves and loads the trained model (and the exact feature columns) so predictions are consistent.
- Preprocessing can align features (dummy columns) to the model's expected feature set.
- Clearer CLI options: --mode (auto/train/predict) and --model path.
- Fails with helpful messages rather than crashing when model/file missing.
"""
import argparse
import os
import sys
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

try:
    import joblib
except Exception:
    # joblib is a lightweight dependency; if not available, ask the user to install it.
    print("The 'joblib' package is required to save/load models. Install with: pip install joblib")
    raise

# Default model filename
DEFAULT_MODEL_PATH = "titanic_model.joblib"

# -------- Helper functions --------
def load_data(path: str) -> pd.DataFrame:
    """Load CSV into a pandas DataFrame. Exit with a user-friendly message if missing."""
    if not os.path.exists(path):
        print(f"Error: file not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df

def preprocess(
    df: pd.DataFrame,
    for_training: bool = True,
    expected_feature_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Selects a few columns and converts them into numeric features.

    If for_training is True, expects 'Survived' to be present and returns (X, y).
    If for_training is False, returns (X, None) and aligns X columns to expected_feature_columns
    by adding missing columns with zero and dropping extra columns.

    Steps:
      - Keep the features we care about
      - Fill missing Age/Fare with the median (simple and easy to understand)
      - Convert Sex to numbers (male -> 0, female -> 1)
      - Fill missing Embarked values and convert to dummy columns
    """
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    df = df.copy()

    # If training, ensure the 'Survived' column exists
    if for_training:
        if 'Survived' not in df.columns:
            raise ValueError("Input CSV must contain a 'Survived' column for training/evaluation.")
        df = df[features + ['Survived']]
    else:
        # For prediction, keep only features that exist in the CSV (we'll fill whatever is missing later)
        # If some feature is missing entirely (e.g. Fare), create a column with NaN so medians can be filled.
        for col in features:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[features]

    # Numeric missing values -> fill with median (computed per-column on the provided df)
    for col in ['Age', 'Fare']:
        median = df[col].median() if not df[col].dropna().empty else 0.0
        df[col] = df[col].fillna(median)
        # Only print in training or when the column existed in the input
        print(f"Filled missing {col} with median = {median}")

    # Sex: map to 0/1; treat unknown as -1
    df['Sex'] = df['Sex'].astype(str).str.lower().map({'male': 0, 'female': 1})
    df['Sex'] = df['Sex'].fillna(-1).astype(int)

    # Embarked: fill missing and convert to dummy columns
    df['Embarked'] = df['Embarked'].fillna('Unknown').astype(str)
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df.drop(columns=['Embarked']), embarked_dummies], axis=1)

    # Final features and target
    if for_training:
        X = df.drop(columns=['Survived'])
        y = df['Survived']
    else:
        X = df
        y = None

    # Align columns if an expected_feature_columns list is provided (useful for ensuring
    # the prediction input has the same columns/order as the model was trained on)
    if expected_feature_columns is not None:
        # Add missing columns with zeros
        for col in expected_feature_columns:
            if col not in X.columns:
                X[col] = 0
        # Drop any extra columns not expected
        extra_cols = [c for c in X.columns if c not in expected_feature_columns]
        if extra_cols:
            X = X.drop(columns=extra_cols)
        # Reorder to match expected order
        X = X[expected_feature_columns]

    print("Preprocessing complete. Feature columns:", list(X.columns))
    return X, y

def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    retrain_on_full: bool = True
) -> RandomForestClassifier:
    """
    Train a RandomForest on a train split, evaluate on test split, print accuracy,
    then retrain on full data for final predictions (if retrain_on_full=True).
    Returns the model trained on the FULL dataset (use this to predict for all rows).
    """
    if test_size > 0 and 0.0 < test_size < 1.0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"Split data: train={len(X_train)} rows, test={len(X_test)} rows")

        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Test accuracy: {acc:.4f}")
    else:
        print("Skipping train/test split (test_size <= 0). Training on full data for initial model.")
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        model.fit(X, y)

    # Retrain on full data if requested (keeps behavior from original script)
    if retrain_on_full:
        model_full = RandomForestClassifier(n_estimators=100, random_state=random_state)
        model_full.fit(X, y)
        print("Retrained model on the full dataset for final predictions.")
        return model_full
    else:
        return model

def save_model(model: RandomForestClassifier, feature_columns: List[str], model_path: str) -> None:
    """Save the model and the expected feature columns to disk together."""
    payload = {
        "model": model,
        "feature_columns": list(feature_columns),
    }
    joblib.dump(payload, model_path)
    print(f"Saved model and feature list to {model_path}")

def load_model(model_path: str):
    """Load model payload saved with save_model(). Returns dict with keys 'model' and 'feature_columns'."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    payload = joblib.load(model_path)
    if not isinstance(payload, dict) or 'model' not in payload or 'feature_columns' not in payload:
        raise ValueError(f"Model file {model_path} has an unexpected format.")
    print(f"Loaded model from {model_path}")
    return payload

def save_predictions(df_original: pd.DataFrame, X: pd.DataFrame, model: RandomForestClassifier, out_path: str) -> pd.DataFrame:
    """Add probability and simple label to original DataFrame and save."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]  # probability of surviving
    else:
        # Fallback: use predict output as 0/1
        preds = model.predict(X)
        probs = preds.astype(float)
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

    # Determine mode if 'auto'
    mode = args.mode.lower()
    if mode == 'auto':
        mode = 'train' if 'Survived' in df.columns else 'predict'
        print(f"Auto mode selected. Determined mode = '{mode}' based on presence of 'Survived' column.")

    model_path = args.model

    if mode == 'train':
        # Training flow: require 'Survived' column
        if 'Survived' not in df.columns:
            print("Error: training mode requires the input CSV to contain a 'Survived' column.")
            sys.exit(1)

        X, y = preprocess(df, for_training=True)
        model = train_and_evaluate(X, y, test_size=args.test_size, random_state=args.random_state)
        # Save model and feature columns so predictions on other CSVs align
        save_model(model, X.columns.tolist(), model_path)

        # Also save predictions on the training CSV if requested
        df_with_preds = save_predictions(df, X, model, args.output)

        print("\nTop 5 predicted probabilities on the input:")
        print(df_with_preds[['Survival_Probability', 'Prediction']].head(5))
        print("\nBottom 5 predicted probabilities on the input:")
        print(df_with_preds[['Survival_Probability', 'Prediction']].tail(5))

    elif mode == 'predict':
        # Prediction flow: load model and apply to input
        try:
            payload = load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            # If the user provided a CSV that does have Survived we can train a new model and save it
            if 'Survived' in df.columns:
                print("Input CSV contains 'Survived'; training a new model and saving it.")
                X_train, y_train = preprocess(df, for_training=True)
                model = train_and_evaluate(X_train, y_train, test_size=args.test_size, random_state=args.random_state)
                save_model(model, X_train.columns.tolist(), model_path)
                payload = {"model": model, "feature_columns": X_train.columns.tolist()}
            else:
                print("No model available and input CSV doesn't contain 'Survived'. Cannot proceed with prediction.")
                sys.exit(1)

        model = payload['model']
        feature_columns = payload['feature_columns']

        # Preprocess for prediction and align columns
        X_pred, _ = preprocess(df, for_training=False, expected_feature_columns=feature_columns)
        df_with_preds = save_predictions(df, X_pred, model, args.output)

        print("\nTop 5 predicted probabilities on the input:")
        print(df_with_preds[['Survival_Probability', 'Prediction']].head(5))
        print("\nBottom 5 predicted probabilities on the input:")
        print(df_with_preds[['Survival_Probability', 'Prediction']].tail(5))

    else:
        print(f"Unknown mode: {args.mode}. Choose from 'auto', 'train', or 'predict'.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Titanic survival prediction (beginner-friendly, now independent).")
    parser.add_argument("--input", default="Titanic-Dataset.csv", help="Input CSV file.")
    parser.add_argument("--output", default="Titanic_Predictions.csv", help="Where to save the predictions CSV.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for the test split (0.0 - 1.0).")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help=f"Path to save/load model (default: {DEFAULT_MODEL_PATH}).")
    parser.add_argument("--mode", default="auto", choices=["auto", "train", "predict"],
                        help="Mode of operation: 'auto' (default) chooses train if Survived present otherwise predict; 'train' to train/save model; 'predict' to load model and predict.")
    args = parser.parse_args()

    main(args)
