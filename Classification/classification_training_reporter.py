import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score


class TrainingReporter:
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self):
        print("Start training...")
        self.model.fit(self.X_train, self.y_train)
        train_pred = self.model.predict(self.X_train)
        train_pred_proba = self.model.predict_proba(self.X_train)
        test_pred = self.model.predict(self.X_test)
        test_pred_proba = self.model.predict_proba(self.X_test)

        train_accuracy = accuracy_score(self.y_train, train_pred)
        train_f1 = f1_score(self.y_train, train_pred, average="weighted")
        train_auroc = roc_auc_score(self.y_train, train_pred_proba, multi_class="ovr", average="weighted")

        test_accuracy = accuracy_score(self.y_test, test_pred)
        test_f1 = f1_score(self.y_test, test_pred, average="weighted")
        test_auroc = roc_auc_score(self.y_test, test_pred_proba, multi_class="ovr", average="weighted")

        print("Training finished!")
        print("Train Accuracy: {:.4f}  |  Test Accuracy: {:.4f}".format(train_accuracy, test_accuracy))
        print("Train F1:       {:.4f}  |  Test F1:       {:.4f}".format(train_f1, test_f1))
        print("Train AUROC:    {:.4f}  |  Test AUROC:    {:.4f}".format(train_auroc, test_auroc))
        print("---------------------------------------------------")
        return self

    def run_cross_validation(self, cv=5):
        print("Start cross validation...")
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        train_accuracy_scores = []
        train_f1_scores = []
        val_accuracy_scores = []
        val_f1_scores = []
        recall_scores = [[], [], []]

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):
            model = self.model.get_new_instance()
            X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_fold_train, y_fold_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            model.fit(X_fold_train, y_fold_train)

            train_predictions = model.predict(X_fold_train)
            val_predictions = model.predict(X_fold_val)
            train_f1 = f1_score(y_fold_train, train_predictions, average="weighted")
            train_accuracy = accuracy_score(y_fold_train, train_predictions)
            val_f1 = f1_score(y_fold_val, val_predictions, average="weighted")
            val_accuracy = accuracy_score(y_fold_val, val_predictions)
            recall = recall_score(y_fold_val, val_predictions, average=None)
            for i, r in enumerate(recall):
                recall_scores[i].append(r)
            train_f1_scores.append(train_f1)
            train_accuracy_scores.append(train_accuracy)
            val_f1_scores.append(val_f1)
            val_accuracy_scores.append(val_accuracy)
            print("Fold {}:".format(fold_idx))
            print("  Train Accuracy: {:.4f}  |  Val Accuracy: {:.4f}".format(train_accuracy, val_accuracy))
            print("  Train F1:       {:.4f}  |  Val F1:       {:.4f}".format(train_f1, val_f1))

            # print("Fold {}: F1 score {:.4f}, Accuracy {:.4f}".format(fold_idx, f1, accuracy))
            # print(classification_report(y_fold_val, predictions))
            print("---------------------------------------------------")

        print("Train Accuracy: {:.4f} +- {:.4f}  |  Val Accuracy: {:.4f} +- {:.4f}".format(
            np.mean(train_accuracy_scores), np.std(train_accuracy_scores),
            np.mean(val_accuracy_scores), np.std(val_accuracy_scores)))
        print("Train F1:       {:.4f} +- {:.4f}  |  Val F1:       {:.4f} +- {:.4f}".format(
            np.mean(train_f1_scores), np.std(train_f1_scores),
            np.mean(val_f1_scores), np.std(val_f1_scores)))
        for i, scores in enumerate(recall_scores):
            print("Recall class {}: {:.4f} +- {:.4f}".format(i, np.mean(scores), np.std(scores)))
        print("Cross validation finished!")
        print("---------------------------------------------------")

    def run_randomized_search_lr(self, cv=5):
        print("Start randomized grid search for Logistic Regression...")

        random_grid_params = {
            'C': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0],
            'l1_ratio': np.linspace(0.0, 1.0, 10),
        }

        model = self.model.get_new_instance().model

        random_grid = RandomizedSearchCV(
            estimator=model,
            param_distributions=random_grid_params,
            cv=cv,
            scoring='f1_weighted',
            n_iter=50,
            random_state=42
        )
        random_grid.fit(self.X_train, self.y_train)

        print("Randomized search finished!")
        print('Best params: {}'.format(random_grid.best_params_))
        print('Best F1 score: {}'.format(random_grid.best_score_))
        print("---------------------------------------------------")

        return random_grid

    def run_randomized_search_rf(self, cv=5):
        print("Start randomized grid search for Random Forest...")

        random_grid_params = {
            'n_estimators': np.arange(100, 1001, 100),
            'max_depth': [3, 4, 5, 6, 7, 8, None],
            'min_samples_split': np.arange(2, 11),
            'min_samples_leaf': np.arange(1, 11),
            'max_features': ['sqrt', 'log2', 0.3, 0.5],
        }

        model = self.model.get_new_instance().model

        random_grid = RandomizedSearchCV(
            estimator=model,
            param_distributions=random_grid_params,
            cv=cv,
            scoring='f1_weighted',
            n_iter=50,
            random_state=42
        )
        random_grid.fit(self.X_train, self.y_train)

        print("Randomized search finished!")
        print('Best params: {}'.format(random_grid.best_params_))
        print('Best F1 score: {}'.format(random_grid.best_score_))
        print("---------------------------------------------------")

        return random_grid

    def run_randomized_search_svc(self, cv=5):
        print("Start randomized grid search for SVC...")

        random_grid_params = {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'kernel': ['rbf', 'linear', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'degree': [2, 3, 4],
        }

        model = self.model.get_new_instance().model

        random_grid = RandomizedSearchCV(
            estimator=model,
            param_distributions=random_grid_params,
            cv=cv,
            scoring='f1_weighted',
            n_iter=50,
            random_state=42
        )
        random_grid.fit(self.X_train, self.y_train)

        print("Randomized search finished!")
        print('Best params: {}'.format(random_grid.best_params_))
        print('Best F1 score: {}'.format(random_grid.best_score_))
        print("---------------------------------------------------")

        return random_grid

    def run_randomized_search_xbg(self, cv=5):
        print("Start randomized grid search for XBGClassifier...")

        n_estimators_params = np.arange(100, 2001, 200)
        learning_rate_params = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15]
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
                                         cv=cv, scoring='f1_weighted', n_iter=125, random_state=42, n_jobs=-1)
        random_grid.fit(self.X_train, self.y_train)

        print("Randomized search finished!")
        print('Best params: {}'.format(random_grid.best_params_))
        print('Best F1 score: {}'.format(random_grid.best_score_))
        print("---------------------------------------------------")

        return random_grid
