
# train.py
import argparse

from src.my_package.models.xgboost_model import XGBoostModel
from src.my_package.models.random_forest_model import RandomForestModel
import pandas as pd
from pathlib import Path
from src.my_package.models.logistic_model import LogisticRegressionModel  # import your model class
import sys

# ----------------------------
# Helper functions
# ----------------------------

def load_train_split(train_dir):
    X_train = pd.read_csv(Path(train_dir) / "X_train.csv")
    y_train = pd.read_csv(Path(train_dir) / "y_train.csv").squeeze()
    return X_train, y_train


def load_transformers(transformer_dir):
    import pickle
    num_cat_path = Path(transformer_dir) / "num_cat_transformer.pkl"
    bins_path = Path(transformer_dir) / "bins_transformer.pkl"
    with open(num_cat_path, 'rb') as f:
        num_cat_trans = pickle.load(f)
    with open(bins_path, 'rb') as f:
        bins_trans = pickle.load(f)
    return num_cat_trans, bins_trans


# ----------------------------
# Main script
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Titanic model with saved transformers")
    parser.add_argument("--train_dir", type=str, required=True, help="Directory where training split CSVs are saved")
    parser.add_argument("--transformer_dir", type=str, required=True, help="Directory where transformers are saved")
    parser.add_argument("--output_model", type=str, default="models/pipeline_model.pkl",
                        help="Path to save trained pipeline")
    args = parser.parse_args()

    print("Loading training split...")
    X_train, y_train = load_train_split(args.train_dir)
    print(f"Training data shape: {X_train.shape}")

    print("Loading saved transformers...")
    num_cat_trans, bins_trans = load_transformers(args.transformer_dir)

    print("Training model...")
    model = XGBoostModel(num_cat_trans, bins_trans, args.output_model)
    model.train(X_train, y_train)


if __name__ == "__main__":
    main()




# python -m scripts.train --train_dir data/train --transformer_dir data/transformers --output_model data/models/pipeline_model.pkl