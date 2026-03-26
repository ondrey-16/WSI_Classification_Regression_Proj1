import numpy as np
import os
import joblib
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from Regression.training_model import CustomXGBRegressorModel

class TrainingReporter:
    def __init__(self, model: CustomXGBRegressorModel, X_train, X_test, y_train, y_test):
        self.model = model
        self.original_model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def run_cross_validation(self, cv=5):
        print("Start cross validation...")
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)

        rmse_scores = []
        mse_scores = []
        r_squared_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(self.X_train)):
            model = self.original_model.get_new_instance()
            X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_fold_train, y_fold_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            model.fit(X_fold_train, y_fold_train)

            predictions = model.predict(X_fold_val)
            mse = mean_squared_error(y_fold_val, predictions)
            mse_scores.append(mse)
            rmse = np.sqrt(mse)
            rmse_scores.append(rmse)
            r_squared = r2_score(y_fold_val, predictions)
            r_squared_scores.append(r_squared)

            print("Fold {}: RMSE = {}, MSE = {}, R^2 = {}".format(fold_idx, rmse, mse, r_squared))

        mean_rmse = np.mean(rmse_scores)
        std_rmse = np.std(rmse_scores)
        mean_mse = np.mean(mse_scores)
        std_mse = np.std(mse_scores)
        mean_r_squared = np.mean(r_squared_scores)
        std_r_squared = np.std(r_squared_scores)

        print("RMSE for cross validation: {} +- {}".format(mean_rmse, std_rmse))
        print("MSE for cross validation: {} +- {}".format(mean_mse, std_mse))
        print("R^2 for cross validation: {} +- {}".format(mean_r_squared, std_r_squared))
        print("Full training:")

        final_model = self.original_model.get_new_instance()
        final_model.fit(self.X_train, self.y_train)
        self.model = final_model

        print("Cross validation finished!")
        print("---------------------------------------------------")
        self.save_model('CV_Model.pkl')

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

        model = self.original_model.get_new_instance().model
        grid = GridSearchCV(model, param_grid=grid_params, cv=cv,
                            scoring= {
                                    'rmse': 'neg_root_mean_squared_error',
                                    'mse': 'neg_mean_squared_error',
                                    'r2': 'r2'},
                            refit="rmse",
                            n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        self.model = grid.best_estimator_

        print("Grid finished!")
        print('Best params: {}'.format(grid.best_params_))
        print('Best RMSE score: {}'.format(-grid.cv_results_['mean_test_rmse'][grid.best_index_]))
        print('Best MSE score: {}'.format(-grid.cv_results_['mean_test_mse'][grid.best_index_]))
        print('Best R^2 score: {}'.format(grid.cv_results_['mean_test_r2'][grid.best_index_]))
        print("---------------------------------------------------")
        self.save_model('GS_Model.pkl')

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

        model = self.original_model.get_new_instance().model
        random_grid = RandomizedSearchCV(estimator=model, param_distributions=random_grid_params,
                                         cv=cv, scoring= {
                                                    'rmse': 'neg_root_mean_squared_error',
                                                    'mse': 'neg_mean_squared_error',
                                                    'r2': 'r2'},
                                         refit="rmse",
                                         n_iter=125, random_state=42, n_jobs=-1)
        random_grid.fit(self.X_train, self.y_train)
        self.model = random_grid.best_estimator_

        print("Randomized search finished!")
        print('Best params: {}'.format(random_grid.best_params_))
        print('Best RMSE score: {}'.format(-random_grid.cv_results_['mean_test_rmse'][random_grid.best_index_]))
        print('Best MSE score: {}'.format(-random_grid.cv_results_['mean_test_mse'][random_grid.best_index_]))
        print('Best R^2 score: {}'.format(random_grid.cv_results_['mean_test_r2'][random_grid.best_index_]))
        print("---------------------------------------------------")
        self.save_model('RGS_Model.pkl')

    def save_model(self, model_filename):
        os.makedirs('saved_models', exist_ok=True)
        path = os.path.join('saved_models', model_filename)
        joblib.dump(self.model, path)
        print("Model saved in: {}".format(path))

    def save_test_set(self):
        os.makedirs('saved_models', exist_ok=True)
        path = os.path.join('saved_models', 'test_set.csv')
        df = self.X_test.copy()
        df["target"] = self.y_test
        df.to_csv(path, index=False)
        print("Test set saved in: {}".format(path))