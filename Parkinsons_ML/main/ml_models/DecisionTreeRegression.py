
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score

class DecisionTreeRegressionCalculator:

    def __init__(self):
        self.tree_reg = DecisionTreeRegressor()
        return

    def train_model(self, train_set, train_labels, test_set, test_labels):
        self.dec_tree_reg_fit(train_set, train_labels)
        self.calculate_rmse(train_set, train_labels)
        # self.predict_values(train_set[:10], train_labels[:10])
        self.predict_values(test_set, test_labels)
        return


    def dec_tree_reg_fit(self, train_set, train_labels):
        self.tree_reg.fit(train_set, train_labels)
        return

    def predict_values(self, test_set, test_labels):
        pred_values = self.tree_reg.predict(test_set)

        list_test_labels = list(test_labels)
        for index in range(len(pred_values)):
            print("Actual_value: {} \t Prediction: {}".format(list_test_labels[index], pred_values[index]))
        return

    def calculate_rmse(self, train_set, train_labels):
        train_predictions = self.tree_reg.predict(train_set)
        tree_mse = mean_squared_error(train_labels, train_predictions)
        tree_rmse = np.sqrt(tree_mse)
        print("Tree Root Mean Square Error: ", tree_rmse)

        scores = cross_val_score(self.tree_reg, train_set, train_labels, scoring="neg_mean_squared_error", cv=10)
        tree_rmse_scores = np.sqrt(-scores)
        self.display_scores(tree_rmse_scores)


    def display_scores(self, scores):
        print("Scores:", scores)
        print("Mean:", scores.mean())
        print("Standard deviation:", scores.std())








