# predict.py
import argparse
import pickle
from pathlib import Path
import pandas as pd

def load_model(model_path: Path):
    """Load trained pipeline from pickle file."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def load_features(features_path: Path):
    """Load features for prediction (no labels)."""
    X = pd.read_csv(features_path)
    return X

def save_predictions(preds, output_path: Path):
    """Save predictions to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction": preds}).to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Predict using a trained pipeline model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model (pickle)")
    parser.add_argument("--features_path", type=str, required=True, help="Path to features CSV file (no labels)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save predictions CSV")
    args = parser.parse_args()

    model = load_model(Path(args.model_path))
    X = load_features(Path(args.features_path))

    print(f"Running inference on {X.shape[0]} samples...")
    preds = model.predict(X)

    save_predictions(preds, Path(args.output_path))

if __name__ == "__main__":
    main()


# python scripts/predict.py --model_path "data/models/pipeline_model.pkl" --features_path "data/featurized/test_features.csv" --output_path "data/prediction/predictions.csv"