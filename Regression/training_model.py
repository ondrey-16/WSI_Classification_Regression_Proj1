from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from Regression.dataset_preprocessing import make_preprocessing_pipeline

class CustomXGBRegressorModel :
    def __init__(self, n_estimators=1000, learning_rate=0.05, max_depth=3, min_child_weight=3,
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1):
        self.model = Pipeline([
        ("preprocessing", make_preprocessing_pipeline()),
        ("model", XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
            )),
        ])
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

        self.is_fitted = False

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_new_instance(self):
        return CustomXGBRegressorModel(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                       max_depth=self.max_depth, min_child_weight=self.min_child_weight,
                                       subsample=self.subsample, colsample_bytree=self.colsample_bytree,
                                       reg_alpha=self.reg_alpha, reg_lambda=self.reg_lambda)

def build_model_from_grid_params(params):
    model_params = {
        key.replace('model__', ''): value
        for key, value in params.items()
    }
    return CustomXGBRegressorModel(**model_params)