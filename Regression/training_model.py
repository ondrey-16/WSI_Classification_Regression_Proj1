from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from Regression.dataset_preprocessing import make_preprocessing_pipeline

class CustomXGBRegressorModel :
    def __init__(self):
        self.model = Pipeline([
        ("preprocessing", make_preprocessing_pipeline()),
        ("model", XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.9,
            reg_lambda=0.1,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
            )),
        ])
        self.is_fitted = False

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_new_instance(self):
        return CustomXGBRegressorModel()