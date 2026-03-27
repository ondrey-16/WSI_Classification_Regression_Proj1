from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from classification_dataset_preprocessing import make_training_pipeline

class CustomXGBClassifierModel :
    def __init__(self, n_estimators=500, learning_rate=0.05, max_depth=2, min_child_weight=5,
                 subsample=0.7, colsample_bytree=0.7, reg_alpha=0.2, reg_lambda=0.1):
        self.model = Pipeline([
        ("model", XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
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

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_new_instance(self):
        return CustomXGBClassifierModel(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                       max_depth=self.max_depth, min_child_weight=self.min_child_weight,
                                       subsample=self.subsample, colsample_bytree=self.colsample_bytree,
                                       reg_alpha=self.reg_alpha, reg_lambda=self.reg_lambda)

def build_model_from_grid_params(params):
    model_params = {
        key.replace('model__', ''): value
        for key, value in params.items()
    }
    return CustomXGBClassifierModel(**model_params)