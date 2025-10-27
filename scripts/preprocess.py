# scripts/preprocess.py
import argparse
from pathlib import Path
import warnings
import pandas as pd
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from my_package.preprocessing.dataloader import DataLoader
from my_package.preprocessing.preprocessor import Preprocess
from my_package.preprocessing.splitting import Splitting

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Titanic dataset")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--output_train", type=str, required=True)
    parser.add_argument("--output_test", type=str, required=True)
    args = parser.parse_args()

    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_test).parent.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    loader = DataLoader()
    train, test = loader.load(args.train_path, args.test_path)
    print(f"Loaded train: {train.shape}, test: {test.shape}")

    print("Cleaning data...")
    preprocessor = Preprocess()
    df = preprocessor.process(train, test)

    print("Splitting data...")
    splitter = Splitting()
    train_processed, test_processed = splitter.split(df)

    print("Saving preprocessed data...")
    train_processed.to_csv(args.output_train, index=False)
    test_processed.to_csv(args.output_test, index=False)

    print(f"Preprocessed train saved to: {args.output_train}")
    print(f"Preprocessed test saved to: {args.output_test}")
    print(f"Final train shape: {train_processed.shape}")
    print(f"Final test shape: {test_processed.shape}")


if __name__ == "__main__":
    main()



# python scripts/preprocess.py --train_path data/titanic/train.csv --test_path data/titanic/test.csv --output_train data/titanic/processed/train_processed.csv --output_test data/titanic/processed/test_processed.csv




