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
from dataset import dataset

from divers.analyse import to_analyze

# This class implements a neural network. The neural_network is trained and tested with an instance of dataset
# This class allows to modify the neural network
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
    def evaluate(self, dataset_instance):
        """ Evaluates the network with the dataset. Arguments : class dataset Out : class to_analyze"""
        history =  self.model.evaluate(dataset_instance.get_RHS(), dataset_instance.get_solutions())
        object_to_analyze = to_analyze(dataset_instance.get_solutions, self.predict(dataset_instance))
        object_to_analyze.add_learning_history(history)
        return object_to_analyze
    def predict(self, dataset_instance):
        assert self.is_compiled
        return self.model.predict(dataset_instance.get_RHS())
    def train_with(self, dataset_instance, epochs,  validation_split = 0):
        """ Trains the network using the dataset. Arguments : class dataset Out : class to_analyze"""
        self.compile()
        assert isinstance(dataset_instance, dataset)
        history = self.fit(dataset_instance.get_RHS, dataset_instance.get_solutions(), epochs, validation_split)
        object_to_analyze =  to_analyze(dataset_instance.get_solutions, self.predict(dataset_instance))
        object_to_analyze.add_learning_history(history)
        return object_to_analyze
