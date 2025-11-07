
# models/base_model.py
from abc import ABC, abstractmethod
import pandas as pd

class BaseModel(ABC):
    """Abstract base class for ML models."""

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the model on features X and target y."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions for the given features X."""
        pass

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate the model and return a metric (e.g., accuracy)."""
        pass