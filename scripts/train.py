# train.py
import pandas as pd
import pickle
import argparse
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline

def load_train_split(train_dir):
    X_train = pd.read_csv(Path(train_dir) / "X_train.csv")
    y_train = pd.read_csv(Path(train_dir) / "y_train.csv").squeeze()  
    return X_train, y_train

def load_transformers(transformer_dir):
    num_cat_path = Path(transformer_dir) / "num_cat_transformer.pkl"
    bins_path = Path(transformer_dir) / "bins_transformer.pkl"
    with open(num_cat_path, 'rb') as f:
        num_cat_trans = pickle.load(f)
    with open(bins_path, 'rb') as f:
        bins_trans = pickle.load(f)
    return num_cat_trans, bins_trans

def create_pipeline(algo, num_cat_trans, bins_trans):
    return Pipeline([
        ('num_cat_transformation', num_cat_trans),
        ('bins', bins_trans),
        ('classifier', algo)
    ])

def save_pipeline(pipeline, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Pipeline saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Train Titanic model with saved transformers")
    parser.add_argument("--train_dir", type=str, required=True, help="Directory where training split CSVs are saved")
    parser.add_argument("--transformer_dir", type=str, required=True, help="Directory where transformers are saved")
    parser.add_argument("--output_model", type=str, default="models/pipeline_model.pkl", help="Path to save trained pipeline")
    args = parser.parse_args()

    print("Loading training split...")
    X_train, y_train = load_train_split(args.train_dir)
    print(f"Training data shape: {X_train.shape}")

    print("Loading saved transformers...")
    num_cat_trans, bins_trans = load_transformers(args.transformer_dir)

    print("Creating pipeline with LogisticRegressionCV...")
    pipeline = create_pipeline(LogisticRegressionCV(cv=5, max_iter=1000), num_cat_trans, bins_trans)

    print("Fitting pipeline on training data...")
    pipeline.fit(X_train, y_train)

    print("Saving pipeline...")
    save_pipeline(pipeline, args.output_model)

if __name__ == "__main__":
    main()




# python scripts/train.py --train_dir data/featurized/train --transformer_dir data/transformers --output_model data/models/pipeline_model.pkl
