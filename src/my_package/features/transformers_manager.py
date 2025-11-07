from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, KBinsDiscretizer
from pathlib import Path
import pickle

class TransformersManager:
    """
    Build, fit, and save transformers — same logic as in featurize.py
    """

    def __init__(self):
        pass

    def build(self):
        num_cat_transformation = ColumnTransformer([
            ('scaling', MinMaxScaler(), [0, 2]),
            ('onehotencolding1', OneHotEncoder(), [1, 3]),
            ('ordinal', OrdinalEncoder(), [4]),
            ('onehotencolding2', OneHotEncoder(), [5, 6])
        ], remainder='passthrough')

        bins = ColumnTransformer([
            ('kbins', KBinsDiscretizer(n_bins=15, encode='ordinal', strategy='quantile'), [0, 2])
        ], remainder='passthrough')

        return num_cat_transformation, bins

    def fit_and_save(self, X_train, num_cat_transformation, bins, output_dir="data/transformers"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        num_cat_transformation.fit(X_train)
        bins.fit(X_train)

        with open(Path(output_dir) / "num_cat_transformer.pkl", "wb") as f:
            pickle.dump(num_cat_transformation, f)

        with open(Path(output_dir) / "bins_transformer.pkl", "wb") as f:
            pickle.dump(bins, f)

        
