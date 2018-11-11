


import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score


class RandomForestRegressionCalculator:

    def __init__(self):
        self.rf_reg = RandomForestRegressor()
        return

    def train_model(self, train_set, train_labels, test_set, test_labels):
        self.rf_reg_fit(train_set, train_labels)
        self.calculate_rmse(train_set, train_labels)
        # self.predict_values(train_set[:10], train_labels[:10])
        # self.predict_values(test_set, test_labels)
        self.fine_tune_model(train_set, train_labels, test_set, test_labels)
        return


    def rf_reg_fit(self, train_set, train_labels):
        self.rf_reg.fit(train_set, train_labels)
        return

    def predict_values(self, test_set, test_labels):
        pred_values = self.rf_reg.predict(test_set)

        list_test_labels = list(test_labels)
        for index in range(len(pred_values)):
            print("Actual_value: {} \t Prediction: {}".format(list_test_labels[index], pred_values[index]))
        return

    def calculate_rmse(self, train_set, train_labels):
        train_predictions = self.rf_reg.predict(train_set)
        tree_mse = mean_squared_error(train_labels, train_predictions)
        tree_rmse = np.sqrt(tree_mse)
        print("Tree Root Mean Square Error: ", tree_rmse)

        scores = cross_val_score(self.rf_reg, train_set, train_labels, scoring="neg_mean_squared_error", cv=10)
        tree_rmse_scores = np.sqrt(-scores)
        self.display_scores(tree_rmse_scores)

    def display_scores(self, scores):
        print("Scores:", scores)
        print("Mean:", scores.mean())
        print("Standard deviation:", scores.std())

    def fine_tune_model(self, train_set, train_labels, test_set, test_labels):
        print("\nFine Tunning Model: \n")

        pred_values = self.rf_reg.predict(test_set)

        param_grid = [
            {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
            {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
        ]
        forest_reg = RandomForestRegressor()
        grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(train_set, train_labels)

        print(grid_search.best_params_)
        # print(grid_search.best_estimator_)

        # cvres = grid_search.cv_results_
        # for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        #     print(np.sqrt(-mean_score), params)

        final_model = grid_search.best_estimator_

        pred_values_1 = final_model.predict(test_set)

        list_test_labels = list(test_labels)
        for index in range(len(pred_values)):
            print("Actual_value: {} \t NFTPrediction: {:.2f} \t FTPrediction: {:.2f}".format(list_test_labels[index], pred_values[index], pred_values_1[index]))

        return




