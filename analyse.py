import tensorflow as tf
import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from bokeh.io import show, output_notebook
from bokeh.plotting import figure
from bokeh.layouts import row, column
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from NeuralNetwork import nn
from dataset import dataset

# This class is created by an instance of nn
# It is used to analyze the nn object performance on the dataset
class to_analyze:
    def __init__(self, solutions, predictions):# solutions and predictions must be vectors
        assert isinstance(solutions, np.ndarray)
        assert isinstance(predictions, np.ndarray)
        self.solutions = solutions
        self.predictions = predictions
        self.hoped_precision = 1.e-6
        assert len(self.predictions) == len(self.solutions)
    def add_learning_history(self, history):
        self.history = history
    def rate_over_precision(self):
        number_over_precision = 0
        for i in range(len(self.solutions)):
            if abs((self.predictions[i] - self.solutions[i])/self.solutions[i]) > self.hoped_precision:
                number_over_precision += 1
        return number_over_precision/len(self.solutions)
    def precision_histogram(self):
        precision_array = np.empty_like(self.solutions)
        for i in range(len(self.solutions)):
            precision_array[i] = abs((self.predictions[i] - self.solutions[i])/self.solutions[i])
        plt.hist(precision_array, density=True, bins = 50)
        plt.axvline(self.hoped_precision, label = "hoped precision = " + str(self.hoped_precision))
        plt.xlabel("precision")
        plt.ylabel("density")
        plt.title("proportion over precision = ", self.rate_over_precision())
        plt.legend()
    def mean_squared_error(self):
        return abs(self.predictions-self.solutions)



