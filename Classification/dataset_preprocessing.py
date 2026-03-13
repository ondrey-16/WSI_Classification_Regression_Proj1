from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.utils.validation import check_is_fitted
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

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

class BalanceClasses(BaseEstimator, TransformerMixin):
    def __init__(self, target_column: str):
        self.target_column = target_column

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        X = X.copy()
        y = X[self.target_column]
        X = X.drop(columns=[self.target_column])

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        result = pd.DataFrame(X_resampled, columns=X.columns)
        result[self.target_column] = y_resampled
        return result

class ScaleNumeric(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_columns = []
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.num_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.scaler.fit(X[self.num_columns])
        self.fitted_ = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        X = X.copy()
        X[self.num_columns] = self.scaler.transform(X[self.num_columns])
        return X

class EncodeCategorial(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cat_columns = []
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def fit(self, X, y=None):
        self.cat_columns = X.select_dtypes(include=["object"]).columns.tolist()
        self.encoder.fit(X[self.cat_columns])
        self.fitted_ = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        X = X.copy()
        encoded = self.encoder.transform(X[self.cat_columns])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(self.cat_columns), index=X.index)
        X = X.drop(columns=self.cat_columns)
        return pd.concat([X, encoded_df], axis=1)

def make_preprocessing_pipeline() -> Pipeline:
    return Pipeline([
        ("balance_classes", BalanceClasses(target_column="growth direction")),
        ("encode_categorical", EncodeCategorial()),
    ])
