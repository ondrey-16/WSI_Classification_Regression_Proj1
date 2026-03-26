from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from classification_dataset_preprocessing import make_training_pipeline

class CustomRandomForestModel :
    def __init__(self, n_estimators=100, max_depth=5, min_samples_split=2,
                 min_samples_leaf=1, max_features="sqrt"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
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
        return CustomRandomForestModel(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features
        )

def build_model_from_grid_params(params):
    model_params = {
        key.replace('model__', ''): value
        for key, value in params.items()
    }
    return CustomRandomForestModel(**model_params)