from abc import ABC, abstractmethod
import pandas as pd

class BaseFeaturesComputer(ABC):
    """Interface for feature computation."""
    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError