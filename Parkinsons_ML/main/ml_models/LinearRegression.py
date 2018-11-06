
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

class LinearRegressionCalculator:

    def __init__(self):
        self.lin_reg = LinearRegression()
        return

    def train_model(self, train_set, train_labels, test_set, test_labels):
        self.lr_fit(train_set, train_labels)
        self.calculate_rmse(train_set, train_labels)
        # self.predict_values(train_set[:10], train_labels[:10])
        self.predict_values(test_set, test_labels)
        return


    def lr_fit(self, train_set, train_labels):
        self.lin_reg.fit(train_set, train_labels)
        return

    def predict_values(self, test_set, test_labels):
        pred_values = self.lin_reg.predict(test_set)

        list_test_labels = list(test_labels)
        for index in range(len(pred_values)):
            print("Actual_value: {} \t Prediction: {}".format(list_test_labels[index], pred_values[index]))
        return

    def calculate_rmse(self, train_set, train_labels):
        train_predictions = self.lin_reg.predict(train_set)
        lin_mse = mean_squared_error(train_labels, train_predictions)
        lin_rmse = np.sqrt(lin_mse)
        print("Linear Root Mean Square Error: ", lin_rmse)
        scores = cross_val_score(self.lin_reg, train_set, train_labels, scoring="neg_mean_squared_error", cv=10)
        lin_rmse_scores = np.sqrt(-scores)
        self.display_scores(lin_rmse_scores)

    def display_scores(self, scores):
        print("Scores:", scores)
        print("Mean:", scores.mean())
        print("Standard deviation:", scores.std())









