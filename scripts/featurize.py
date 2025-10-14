"""
featurize.py

CLI script to perform feature engineering on the Titanic dataset.
Implements the same feature transformations found in the provided notebook
and writes out processed CSV(s) ready for training/evaluation.

Usage example:
    python featurize.py --train ../data/titanic/train.csv --test ../data/titanic/test.csv --out-dir ../data/processed --split

Options:
    --train    Path to train CSV (required)
    --test     Path to test CSV (required)
    --out-dir  Output directory for processed files (default: ./processed)
    --split    If provided, write processed_train.csv and processed_test.csv separately.
               Otherwise writes processed_combined.csv with a column `is_train` and `Survived` (Survived may be NaN for test rows).
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    logging.info(f"Loaded train: {train.shape}, test: {test.shape}")
    return train, test


def basic_clean(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Drop Cabin because of many missing values
    train = train.copy()
    test = test.copy()
    if 'Cabin' in train.columns:
        train.drop(columns=['Cabin'], inplace=True)
    if 'Cabin' in test.columns:
        test.drop(columns=['Cabin'], inplace=True)

    # Fill obvious nulls like in the notebook
    if 'Embarked' in train.columns:
        train['Embarked'].fillna('S', inplace=True)
    if 'Fare' in test.columns:
        test['Fare'].fillna(test['Fare'].mean(), inplace=True)

    return train, test


def unify_and_fill_age(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    # Create a unified df to do group median fill for Age
    train_idx = train.index
    test_idx = test.index + len(train)  # not necessary but helpful if we need to split later

    combined = pd.concat([train, test], sort=True).reset_index(drop=True)

    # Fill Age by median grouped by Sex and Pclass (same logic as notebook)
    if 'Age' in combined.columns:
        combined['Age'] = combined.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))

    return combined


def extract_title(combined: pd.DataFrame) -> pd.DataFrame:
    # Extract title from Name
    combined = combined.copy()
    combined['Title'] = combined['Name'].str.split(', ').str[1].str.split('.').str[0]

    # Replace rare and alternate titles like in the notebook
    rare_titles = ['Lady', 'the Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    combined['Title'] = combined['Title'].replace(rare_titles, 'Rare')
    combined['Title'] = combined['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

    return combined


def family_size_feature(combined: pd.DataFrame) -> pd.DataFrame:
    combined = combined.copy()
    combined['Family_size'] = combined['SibSp'] + combined['Parch'] + 1

    def family_size_label(number: int) -> str:
        if number == 1:
            return 'Alone'
        elif 1 < number < 5:
            return 'Small'
        else:
            return 'Large'

    combined['Family_size'] = combined['Family_size'].apply(family_size_label)
    return combined


def drop_unused(combined: pd.DataFrame) -> pd.DataFrame:
    combined = combined.copy()
    # Drop columns the notebook dropped
    to_drop = [c for c in ['Name', 'Parch', 'SibSp', 'Ticket'] if c in combined.columns]
    if to_drop:
        combined.drop(columns=to_drop, inplace=True)
    return combined


def final_cleanup(combined: pd.DataFrame) -> pd.DataFrame:
    # Cast Age to int like notebook
    if 'Age' in combined.columns:
        # after filling there might still be NaNs if groups were all NaN; fill with global median then convert
        combined['Age'].fillna(int(combined['Age'].median()), inplace=True)
        combined['Age'] = combined['Age'].astype('int64')

    # Ensure Survived is int where present
    if 'Survived' in combined.columns:
        combined['Survived'] = pd.to_numeric(combined['Survived'], errors='coerce')
        # keep NaN for test set; for train rows it will be converted to int later if needed

    return combined


def split_back(combined: pd.DataFrame, train_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = combined.iloc[:train_rows].reset_index(drop=True)
    test = combined.iloc[train_rows:].reset_index(drop=True)
    return train, test


def write_outputs(combined: pd.DataFrame, out_dir: Path, split: bool, train_rows: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if split:
        train, test = split_back(combined, train_rows)
        # For train, ensure Survived is integer
        if 'Survived' in train.columns:
            train['Survived'] = train['Survived'].astype('int64')
        train.to_csv(out_dir / 'processed_train.csv', index=False)
        test.to_csv(out_dir / 'processed_test.csv', index=False)
        logging.info(f"Wrote processed_train.csv ({train.shape}) and processed_test.csv ({test.shape}) to {out_dir}")
    else:
        # add helper column to identify original rows
        combined['is_train'] = [True] * train_rows + [False] * (combined.shape[0] - train_rows)
        combined.to_csv(out_dir / 'processed_combined.csv', index=False)
        logging.info(f"Wrote processed_combined.csv ({combined.shape}) to {out_dir}")


def featurize(train_path: str, test_path: str, out_dir: str, split: bool = False) -> None:
    train, test = load_data(train_path, test_path)
    train, test = basic_clean(train, test)
    train_rows = train.shape[0]

    combined = unify_and_fill_age(train, test)
    combined = extract_title(combined)
    combined = family_size_feature(combined)
    combined = drop_unused(combined)
    combined = final_cleanup(combined)

    write_outputs(combined, Path(out_dir), split, train_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Featurize Titanic dataset (from notebook)')
    parser.add_argument('--train', required=True, help='Path to train CSV')
    parser.add_argument('--test', required=True, help='Path to test CSV')
    parser.add_argument('--out-dir', default='./processed', help='Output directory')
    parser.add_argument('--split', action='store_true', help='Write separate train/test processed CSVs')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    featurize(args.train, args.test, args.out_dir, args.split)


if __name__ == '__main__':
    main()
