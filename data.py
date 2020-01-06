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


class RHS:# contains a set of second members
    def __init__(self, data):#data can be a file name or an istance of np.array with the solutions
        if isinstance(data, np.matrix) or isinstance(data, np.array): #if it's a matrix
            self.content = data # then we get it
        if isinstance(data, list):
            self.content = np.array(data)

    def normalize_standard(self):
        scaler = StandardScaler()
        self.content = scaler.fit_transform(self.content)
    def max(self):
        """returns an array with the max value for each coordinate"""
        lmax = []
        (n,p) = self.content.shape
        for j in range(p):
            lmax.append(max(self.content[:,j]))
        return np.array(lmax)
    def min(self):
        """returns an array with the min value for each coordinate"""
        lmin = []
        (n,p) = self.content.shape
        for j in range(p):
            lmin.append(min(self.content[:,j]))
        return np.array(lmin)
    def range(self):
        """returns an array with the range for each coordinate"""
        return self.max() - self.min()
    def boxplot_range(self):
        """ boxplot of the range for each coordinate """
        plt.boxplot(self.range(), whis=[2.5,97.5])
    def size(self):
        """number of second members"""
        return self.content.size
    def get_second_members(self):
        return self.content


class solutions:
    def __init__(self, data):
        """data can be a list or an istance of np.array with the solutions"""
        if isinstance(data, np.array) or isinstance(data, list):# if data is a vector
            self.content = np.array(data)# then we get its value
    def size(self):
        """number of solutions"""
        return self.content.size
    def mean(self):
        """mean value of the solutions"""
        return np.mean(self.content)
    def apply(self, f):
        """f is the function applied to every solution"""
        for i in range(self.size()):
            self.content[i] = f(self.content[i])
    def apply_linear(self, a, b):
        """ applies a linear transformation to every solution"""
        self.apply(lambda x: a*x + b)
    def box(self):
        """boxplot of the solutions"""
        plt.boxplot(self.content, whis=[2.5, 97.5])
    def get_solutions(self):
        """return an array with the solutions"""
        return self.content



class dataset:
    def init(self, second_members_anytype, solutions_anytype):
        """ Contains second members and solutions.The type can be string, array or class"""
        if isinstance(second_members_anytype, str) and isinstance(solutions_anytype, str):
            self.second_members = second_members(second_members_anytype)
            self.train_solutions = solutions(solutions_anytype)
        if isinstance(second_members_anytype, second_members) and isinstance(solutions_anytype, solutions):
            self.second_members = second_members_anytype
            self.solutions = solutions_anytype
        if isinstance(second_members_anytype, np.matrix) and isinstance(solutions_anytype, np.array):
            self.second_members = second_members(second_members_anytype)
            self.solutions = solutions(second_members_anytype)
        assert self.solutions.size() == self.second_members.size()
        self.predictions = None # will contain a set of predicted values
        self.used_nn = None
        self.history = None
    def eliminate_under(self, upper_limit):
        """Eliminates the coordinates with range under the upper limit"""
        raise("function not defined yet :)")
    def get_solutions(self):
        """returns the solutions"""
        return self.solutions.get_solutions()
    def get_second_members(self):
        """returns the solutions"""
        return self.second_members.get_second_members()
    def put_through(self, neural_network, epochs, validation_split):
        """Puts the dataset through teh neural network. The predictions are stored in the predictions field."""
        assert isinstance(neural_network, nn)
        neural_network.compile()# compiles the network if it has not already been done
        self.history = neural_network.fit(self.get_second_members(), self.get_solutions(), epochs, validation_split)
    def visualize_model(self):
        """plots the evolution of the model"""
        loss_curve = self.history.history["loss"]
        loss_val_curve = self.history.history["val_loss"]
        plt.plot(loss_curve, label="Train")
        plt.plot(loss_val_curve, label="Val")
        plt.legend(loc='upper left')
        plt.title("Loss")
        plt.show()
    def test(self, neural_network):
        """ returns the test performance of the neural network on the dataset"""
        neural_network.compile()
        return neural_network.evaluate(self.get_second_members(), self.get_solutions())

