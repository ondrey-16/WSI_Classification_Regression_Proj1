from sklearn.pipeline import Pipeline
from Regression.dataset_preprocessing import make_preprocessing_pipeline

class CustomXGBRegressorModel :
    def __init__(self, model):
        self.model = Pipeline([
        ("preprocessing", make_preprocessing_pipeline()),
        ("model", model),
        ])
        self.regression_model = model

        self.is_fitted = False

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_new_instance(self):
        return CustomXGBRegressorModel(model=self.regression_model)