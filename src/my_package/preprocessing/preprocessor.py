from typing import Tuple
import pandas as pd
from .base_preprocessor import BasePreprocessor


class Preprocess(BasePreprocessor):
    """
    Clean data. Mirrors the logic from your script's `clean_data`.
    Methods operate on DataFrames and return the combined DataFrame,
    leaving splitting to the Splitter class.
    """

    def process(self, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        # Operate on copies
        train = train.copy()
        test = test.copy()

        # Drop Cabin (inplace in original script)
        if "Cabin" in train.columns:
            train.drop(columns=["Cabin"], inplace=True)
        if "Cabin" in test.columns:
            test.drop(columns=["Cabin"], inplace=True)

        # Fill Embarked for train
        if "Embarked" in train.columns:
            train["Embarked"].fillna("S", inplace=True)

        # Fill Fare for test (mean)
        if "Fare" in test.columns and test["Fare"].isnull().any():
            test["Fare"].fillna(test["Fare"].mean(), inplace=True)

        # Concatenate to do group Age imputation (same as original)
        df = pd.concat([train, test], sort=True).reset_index(drop=True)

        # Age imputation by Sex & Pclass groups (median)
        if {"Age", "Sex", "Pclass"}.issubset(df.columns):
            df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
                lambda x: x.fillna(x.median())
            )

        return df