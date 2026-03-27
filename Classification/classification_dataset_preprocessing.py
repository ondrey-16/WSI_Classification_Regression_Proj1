from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

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

class StandardizeNumeric(BaseEstimator, TransformerMixin):
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

class LabelEncodeTarget(BaseEstimator, TransformerMixin):
    def __init__(self, target_column):
        self.target_column = target_column
        self.encoder = LabelEncoder()

    def fit(self, X, y=None):
        self.encoder.fit(X[self.target_column])
        self.fitted_ = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        X = X.copy()
        X[self.target_column] = self.encoder.transform(X[self.target_column])
        return X

class ComputeDeltas(BaseEstimator, TransformerMixin):
    def __init__(self, prefix_a="9", prefix_b="12", exclude_columns=None):
        self.prefix_a = prefix_a
        self.prefix_b = prefix_b
        self.exclude_columns = exclude_columns or []

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X, y=None):
        X = X.copy()
        result = {}

        cols_a = {col.removeprefix(f"{self.prefix_a}_"): col
                  for col in X.columns
                  if col.startswith(f"{self.prefix_a}_")}

        cols_b = {col.removeprefix(f"{self.prefix_b}_"): col
                  for col in X.columns
                  if col.startswith(f"{self.prefix_b}_")}

        common_params = set(cols_a.keys()) & set(cols_b.keys())

        for param in common_params:
            result[f"delta_{param}"] = X[cols_b[param]] - X[cols_a[param]]

        for col in self.exclude_columns:
            if col in X.columns:
                result[col] = X[col]

        return pd.DataFrame(result, index=X.index)

class ComputeRelativeDeltas(BaseEstimator, TransformerMixin):
    def __init__(self, prefix_a="9", prefix_b="12", exclude_columns=None):
        self.prefix_a = prefix_a
        self.prefix_b = prefix_b
        self.exclude_columns = exclude_columns or []

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X, y=None):
        X = X.copy()
        result = {}

        cols_a = {col.removeprefix(f"{self.prefix_a}_"): col
                  for col in X.columns
                  if col.startswith(f"{self.prefix_a}_")}

        cols_b = {col.removeprefix(f"{self.prefix_b}_"): col
                  for col in X.columns
                  if col.startswith(f"{self.prefix_b}_")}

        common_params = set(cols_a.keys()) & set(cols_b.keys())

        for param in common_params:
            result[f"delta_{param}"] = (X[cols_b[param]] - X[cols_a[param]])/X[cols_b[param]]

        for col in self.exclude_columns:
            if col in X.columns:
                result[col] = X[col]

        return pd.DataFrame(result, index=X.index)

def make_preprocessing_pipeline() -> Pipeline:
    return Pipeline([
        ("standarize_numeric", StandardizeNumeric()),
        ("encode_labels", LabelEncodeTarget(target_column="growth direction")),
        ("balance_classes", BalanceClasses(target_column="growth direction")),
    ])

def make_delta_pipeline() -> Pipeline:
    return Pipeline([
        ("encode_to_deltas", ComputeDeltas(exclude_columns=["growth direction"])),
    ])

def make_label_pipeline() -> Pipeline:
    return Pipeline([
        ("encode_labels", LabelEncodeTarget(target_column="growth direction")),
    ])

def make_training_pipeline() -> ImbPipeline:
    return ImbPipeline([
        ("standarize_numeric", StandardizeNumeric()),
        ("balance_classes", SMOTE(random_state=42)),
    ])

def make_standarize_pipeline() -> Pipeline:
    return Pipeline([
        ("standarize_numeric", StandardizeNumeric()),
    ])

def make_relative_delta_pipeline() -> Pipeline:
    return Pipeline([
        ("relative deltas", ComputeRelativeDeltas(exclude_columns=["growth direction"]))
    ])