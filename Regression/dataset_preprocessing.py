from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.utils.validation import check_is_fitted
import numpy as np
import pandas as pd

class FillMasVnrTypeNans(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.copy()
        X["MasVnrType"] = X["MasVnrType"].fillna("?")
        return X

class FillNumericValues(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str]):
        self.columns = columns
        self.imputer = KNNImputer(n_neighbors=5)
        self.is_fitted = False
    def fit(self, X, y=None):
        self.imputer.fit(X[self.columns])
        self.fitted_ = True
        return self
    def transform(self, X, y=None):
        check_is_fitted(self)
        X = X.copy()
        X[self.columns] = self.imputer.transform(X[self.columns])
        return X

class SplitToBins(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str], num_bins: list[int]):
        self.columns = columns
        self.num_bins = num_bins
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.copy()
        for i in range(len(self.columns)):
            X[self.columns[i]] = pd.cut(X[self.columns[i]], self.num_bins[i])
        return X

class LogTransform(BaseEstimator, TransformerMixin):
    def __init__(self, log_columns : list):
        self.log_columns = log_columns
        self.transform_output = None
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.copy()
        for col in self.log_columns:
            X[col] = np.log1p(X[col])
        return X

class NumCatTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_columns = []
        self.cat_columns = []
        self.column_transformer = None
        self.is_fitted = False
    def fit(self, X, y=None):
        self.num_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.cat_columns = X.select_dtypes(include=["object"]).columns.tolist()

        self.column_transformer = ColumnTransformer([
            ("num", StandardScaler(), self.num_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_columns),
        ])

        self.column_transformer.fit(X, y)
        self.fitted_ = True
        return self
    def transform(self, X, y=None):
        check_is_fitted(self)
        X = X.copy()
        return self.column_transformer.transform(X)


def make_preprocessing_pipeline() -> Pipeline:
    to_fill_columns = ["MasVnrArea", "LotFrontage", "GarageYrBlt"]
    log_columns = ["LotArea", "BsmtUnfSF"]
    split_to_bins_columns = ["YearBuilt", "YearRemodAdd", "BsmtFinSF2", "GarageYrBlt"]
    num_bins = [6, 6, 3, 6]

    return Pipeline([
        ('fill_mas_vnr_type_nans', FillMasVnrTypeNans()),
        ('fill_num_nans', FillNumericValues(to_fill_columns)),
        ('split_to_bins', SplitToBins(split_to_bins_columns, num_bins)),
        ('log_transform', LogTransform(log_columns)),
        ("num_cat_transform", NumCatTransform()),
    ])
