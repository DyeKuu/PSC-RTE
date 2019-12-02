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


class nn:
    def __init__(self):
        self.model = tf.keras.models.Sequential()
        self.loss = "mean_squared_error"
        self.optimizer = "sgd"
        self.metrics = ["mean_squared_error"]
        self.is_compiled = False # says if self.model already has been compiled with the layers, optmizer, loss and metrics
    def basic_nn(self, list_n_neurons):
        """Initialises the network with layers whose numbers of neurons are given in the argument"""
        assert isinstance(list_n_neurons, list) or isinstance(list_n_neurons, list)
        self.__init__()
        for nb_neurons in list_n_neurons:
            self.model.add(tf.keras.layers.Dense(nb_neurons, activation="relu"))
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    def add_relu(self, nb_neurons):
        self.model.add(tf.keras.layers.Dense(nb_neurons, activation = "relu"))
    def add_sigmoid(self, nb_neurons):
        self.model.add(tf.keras.layers.Dense(nb_neurons, activation = "sigmoid"))
    def compile(self):
        if not(self.is_compiled):
            self.model.compile(loss = self.loss, optimizer = self.optimizer, metrics = self.metrics)
            self.is_compiled = True
    def set_optimizer(self, optimizer_name):
        self.optimizer = optimizer_name
    def set_loss(self, loss_name):
        self.loss = loss_name
    def set_metrics(self, metrics_name):
        self.metrics = [metrics_name]
    def fit(self, pb_train, sol_train, epochs, validation_split):
        return self.model.fit(pb_train, sol_train, epochs, validation_split)
    def evaluate(self, pb_test, sol_test):
        return self.model.evaluate(pb_test, sol_test)