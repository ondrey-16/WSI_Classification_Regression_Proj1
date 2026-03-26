from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from classification_dataset_preprocessing import make_training_pipeline

class CustomSVCModel :
    def __init__(self, C=1.0, kernel="rbf", gamma="scale", degree=3):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree

        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            class_weight="balanced",
            probability=True,  # potrzebne do predict_proba i auroc
            random_state=42
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
        return CustomSVCModel(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree
        )

def build_model_from_grid_params(params):
    model_params = {
        key.replace('model__', ''): value
        for key, value in params.items()
    }
    return CustomSVCModel(**model_params)