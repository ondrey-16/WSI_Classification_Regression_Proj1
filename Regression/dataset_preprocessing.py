from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from abc import ABC, abstractmethod

class BasePipeline(ABC, BaseEstimator, TransformerMixin):
    @abstractmethod
    def fit(self, X, y=None):
        pass
    @abstractmethod
    def transform(self, X, y=None):
        pass

class DropColumns(BasePipeline):
    def __init__(self, drop_columns : list):
        self.drop_columns = drop_columns
        self.transform_output = None
    def set_output(self, *, transform=None):
        self.transform_output = transform
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.copy()
        return X.drop(self.drop_columns, axis=1)

class FillMasVnrTypeNans(BasePipeline):
    def __init__(self):
        self.transform_output = None
    def set_output(self, *, transform=None):
        self.transform_output = transform
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.copy()
        X["MasVnrType"] = X["MasVnrType"].fillna("?")
        return X

class RemoveOutliers(BasePipeline):
    def __init__(self, columns : list[str], ranges: list[int]):
        self.columns = columns
        self.ranges = ranges
        self.transform_output = None
    def set_output(self, *, transform=None):
            self.transform_output = transform
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.copy()
        for i in range(len(self.columns)):
            X = X[X[self.columns[i]] <= self.ranges[i]]
        return X

class LogTransform(BasePipeline):
    def __init__(self, log_columns : list):
        self.log_columns = log_columns
        self.transform_output = None
    def set_output(self, *, transform=None):
            self.transform_output = transform
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.copy()
        for col in self.log_columns:
            X[col] = np.log1p(X[col])
        return X

class NumCatTransform(BasePipeline):
    def __init__(self):
        self.num_columns = []
        self.cat_columns = []
        self.column_transformer = None
        self.transform_output = None
    def set_output(self, *, transform=None):
        self.transform_output = transform
    def fit(self, X, y=None):
        self.num_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()

        self.column_transformer = ColumnTransformer([
            ("num", StandardScaler(), self.num_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_columns),
        ])
        self.column_transformer.set_output(transform=self.transform_output)

        self.column_transformer.fit(X, y)
        return self
    def transform(self, X, y=None):
        X = X.copy()
        return self.column_transformer.transform(X)


def make_pipeline() -> Pipeline:
    drop_columns = ["Id"]
    log_columns = ["LotArea"]
    rem_columns = ["LotArea", "GrLivArea"]
    ranges = [100000, 5000]

    return Pipeline([
        ('drop_columns', DropColumns(drop_columns)),
        ('fill_nans', FillMasVnrTypeNans()),
        ('remove_outliers', RemoveOutliers(rem_columns, ranges)),
        ('log_transform', LogTransform(log_columns)),
        ("num_cat_transform", NumCatTransform()),
    ])
