# models/logistic_model.py
from abc import ABC
from src.my_package.models.base_model import BaseModel
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
import pickle
from pathlib import Path

class LogisticRegressionModel(BaseModel, ABC):
    def __init__(self, num_cat_trans, bins_trans, output_path):
        self.num_cat_trans = num_cat_trans
        self.bins_trans = bins_trans
        self.output_path = output_path
        self.pipeline = None

    def train(self, X, y):
        # This is the content of your train.py logic
        self.pipeline = Pipeline([
            ('num_cat_transformation', self.num_cat_trans),
            ('bins', self.bins_trans),
            ('classifier', LogisticRegressionCV(cv=5, max_iter=1000))
        ])
        self.pipeline.fit(X, y)

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        print(f"Pipeline saved to: {self.output_path}")

    def predict(self, X):
        return self.pipeline.predict(X)

    def evaluate(self, X, y):
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)