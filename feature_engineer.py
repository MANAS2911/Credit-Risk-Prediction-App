# feature_engineer.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['Debt/Duration'] = X['Credit amount'] / (X['Duration'] + 1e-5)
        X['Age_group'] = pd.cut(X['Age'], bins=[18, 30, 45, 60, 100],
                                labels=['18-30', '31-45', '46-60', '60+'])
        return X
