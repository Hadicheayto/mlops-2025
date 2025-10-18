# train.py
"""
Train a simple Logistic Regression model on preprocessed features and save it.
Usage:
    python train.py --input data/titanic/features/features.csv --output models/logistic_model.pkl
"""

import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Train a Logistic Regression model")
    parser.add_argument("--input", required=True, help="Path to CSV with features including target 'Survived'")
    parser.add_argument("--output", required=True, help="Path to save trained model (pickle)")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)
    
    # Separate features and target
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Ensure output folder exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Save model
    with open(args.output, "wb") as f:
        pickle.dump(model, f)

    print(f"Model trained and saved to {args.output}")

if __name__ == "__main__":
    main()
