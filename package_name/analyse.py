import numpy as np
import matplotlib.pyplot as plt


# This class is created by an instance of nn
# It is used to analyze the nn object performance on the dataset
class to_analyze:
    def __init__(self, solutions, predictions):  # solutions and predictions must be vectors
        assert isinstance(solutions, np.ndarray)
        assert isinstance(predictions, np.ndarray)
        self.solutions = solutions
        self.predictions = predictions
        self.hoped_precision = 1.e-6
        self.size = len(self.solutions)
        self.history = None
        self.used_nn = None
        assert len(self.predictions) == len(self.solutions)

    def add_learning_history(self, history):
        self.history = history

    def add_used_nn(self, neural_network):
        self.used_nn = neural_network

    def rate_over_precision(self):
        number_over_precision = 0
        for i in range(len(self.solutions)):
            if abs((self.predictions[i] - self.solutions[i]) / self.solutions[i]) > self.hoped_precision:
                number_over_precision += 1
        print("proportion of predictions over relative precision ", self.hoped_precision, " is ",
              number_over_precision / self.size)
        return number_over_precision / self.size

    def precision_histogram(self):
        precision_array = np.empty_like(self.solutions)
        for i in range(len(self.solutions)):
            precision_array[i] = abs((self.predictions[i] - self.solutions[i]) / self.solutions[i])
        plt.hist(precision_array, density=True, bins=50)
        plt.axvline(self.hoped_precision, label="hoped precision = " + str(self.hoped_precision))
        plt.xlabel("precision")
        plt.ylabel("density")
        plt.title("proportion over precision = " + str(self.rate_over_precision()))
        plt.legend()
        # plt.show()

    def mean_squared_error(self):
        return abs(self.predictions - self.solutions)

    def mean_precision_error(self):
        return np.mean(np.absolute((self.predictions - self.solutions) / self.solutions))
