# scripts/predict.py
import argparse
from pathlib import Path
import pandas as pd
from src.my_package.models.logistic_model import LogisticRegressionModel  # import your model class
import pickle
import sys
# ----------------------------
# Helper functions
# ----------------------------
def load_model(model_path: Path):
    """Load trained pipeline from pickle file."""
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline  # we return the fitted pipeline

def load_features(features_path: Path):
    """Load features for prediction (no labels)."""
    X = pd.read_csv(features_path)
    return X

def save_predictions(preds, output_path: Path):
    """Save predictions to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction": preds}).to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")

# ----------------------------
# Main script
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict using a trained pipeline model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model (pickle)")
    parser.add_argument("--features_path", type=str, required=True, help="Path to features CSV file (no labels)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save predictions CSV")
    args = parser.parse_args()

    # Load features
    X = load_features(Path(args.features_path))
    print(f"Running inference on {X.shape[0]} samples...")

    # Load the trained pipeline directly
    with open(args.model_path, "rb") as f:
        pipeline = pickle.load(f)

    # Create a temporary LogisticRegressionModel instance to use the predict method
    model = LogisticRegressionModel(None, None, None)
    model.pipeline = pipeline
    preds = model.predict(X)

    save_predictions(preds, Path(args.output_path))


if __name__ == "__main__":
    main()


# python -m scripts.predict --model_path "data/models/pipeline_model.pkl" --features_path "data/featurized/test_features.csv" --output_path "data/prediction/predictions.csv"
