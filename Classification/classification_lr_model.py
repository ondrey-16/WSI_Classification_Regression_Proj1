from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


class CustomLogisticRegressionModel:
    def __init__(self, C=1.0, l1_ratio=0.0, solver="saga", max_iter=10000):
        self.C = C
        self.l1_ratio = l1_ratio
        self.solver = solver
        self.max_iter = max_iter

        self.model = LogisticRegression(
            C=C,
            l1_ratio=l1_ratio,
            solver="saga",
            max_iter=max_iter,
            class_weight="balanced",
            random_state=42,
        )

        self.is_fitted = False

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_new_instance(self):
        return CustomLogisticRegressionModel(
            C=self.C,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter
        )

def build_model_from_grid_params(params):
    model_params = {
        key.replace('model__', ''): value
        for key, value in params.items()
    }
    return CustomLogisticRegressionModel(**model_params)
