import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from Regression.training_model import CustomXGBRegressorModel

class TrainingReporter:
    def __init__(self, model: CustomXGBRegressorModel, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self):
        print("Start training...")
        self.model.fit(self.X_train, self.y_train)
        print("Training finished!")
        print("---------------------------------------------------")
        return self

    def run_cross_validation(self, cv=5):
        print("Start cross validation...")
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)

        rmse_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(self.X_train)):
            model = self.model.get_new_instance()
            X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_fold_train, y_fold_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            model.fit(X_fold_train, y_fold_train)

            predictions = model.predict(X_fold_val)
            rmse = np.sqrt(mean_squared_error(y_fold_val, predictions))
            rmse_scores.append(rmse)

            print("Fold {}: RMSE {}".format(fold_idx, rmse))

        mean_rmse = np.mean(rmse_scores)
        std_rmse = np.std(rmse_scores)

        print("RMSE for cross validation: {} +- {}".format(mean_rmse, std_rmse))
        print("Cross validation finished!")
        print("---------------------------------------------------")