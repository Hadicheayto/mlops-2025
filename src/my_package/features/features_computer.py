# src/my_package/features/features_computer.py
import pandas as pd
from .base_features_computer import BaseFeaturesComputer

class FeaturesComputer(BaseFeaturesComputer):
    def extract_title(self, df):
        df = df.copy()
        df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
        df['Title'] = df['Title'].replace(
            ['Lady', 'the Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 
             'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
        return df

    def create_family_size(self, df):
        df = df.copy()
        df['Family_size'] = df['SibSp'] + df['Parch'] + 1
        def family_size_bin(number):
            if number == 1:
                return "Alone"
            elif number < 5:
                return "Small"
            else:
                return "Large"
        df['Family_size'] = df['Family_size'].apply(family_size_bin)
        return df

    def create_group_size(self, df):
        df = df.copy()
        if 'Ticket' in df.columns:
            df['GroupSize'] = df.groupby('Ticket')['Ticket'].transform('count')
        else:
            df['GroupSize'] = 1
        return df

    def drop_unused_columns(self, df):
        df = df.copy()
        df.drop(columns=['Name','Parch','SibSp','Ticket','PassengerId'], inplace=True, errors='ignore')
        return df

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.extract_title(df)
        df = self.create_family_size(df)
        df = self.create_group_size(df)   # must be before dropping Ticket
        df = self.drop_unused_columns(df)
        return df