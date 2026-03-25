import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
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
        train_pred = self.model.predict(self.X_train)
        rmse = np.sqrt(mean_squared_error(train_pred, self.y_train))

        print("Training finished!")
        print("Train RMSE: {}".format(rmse))
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

    def run_grid_search(self, cv=5):
        print("Start grid search...")
        grid_params = {
            'model__n_estimators': [500, 1000, 2000],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [3, 4, 5],
            'model__min_child_weight': [1, 2, 3],
            'model__subsample': [0.6, 0.8],
            'model__colsample_bytree': [0.6, 0.8],
        }

        model = self.model.get_new_instance().model
        grid = GridSearchCV(model, param_grid=grid_params, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
        grid.fit(self.X_train, self.y_train)

        print("Grid finished!")
        print('Best params: {}'.format(grid.best_params_))
        print('Best RMSE score: {}'.format(grid.best_score_))
        print("---------------------------------------------------")

        return grid

    def run_randomized_search(self, cv=5):
        print("Start randomized grid search...")

        n_estimators_params = np.arange(100, 2001, 200)
        learning_rate_params = [0.01, 0.3, 0.05, 0.07, 0.1, 0.15]
        max_depth_params = np.arange(1, 9)
        min_child_weight_params = np.arange(1, 9)
        subsample_params = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        colsample_bytree_params = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        reg_alpha_params = [0, 0.01, 0.1, 0.4, 0.6, 0.8, 1.0, 5.0]
        reg_lambda_params = [0, 0.01, 0.1, 0.4, 0.6, 0.8, 1.0, 5.0, 10.0]

        random_grid_params = {
            'model__n_estimators': n_estimators_params,
            'model__learning_rate': learning_rate_params,
            'model__max_depth': max_depth_params,
            'model__min_child_weight': min_child_weight_params,
            'model__subsample': subsample_params,
            'model__colsample_bytree': colsample_bytree_params,
            'model__reg_alpha': reg_alpha_params,
            'model__reg_lambda': reg_lambda_params,
        }

        model = self.model.get_new_instance().model

        random_grid = RandomizedSearchCV(estimator=model, param_distributions=random_grid_params,
                                         cv=cv, scoring='neg_root_mean_squared_error', n_iter=125, random_state=42, n_jobs=-1)
        random_grid.fit(self.X_train, self.y_train)

        print("Randomized search finished!")
        print('Best params: {}'.format(random_grid.best_params_))
        print('Best RMSE score: {}'.format(random_grid.best_score_))
        print("---------------------------------------------------")

        return random_grid