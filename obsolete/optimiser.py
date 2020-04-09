# This class is used as an optimisor of the neural network, where generally we make use of the package hyperas.

# import hyperas
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

# import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
from package_name import NeuralNetwork as nn
from package_name import dataset

def get_data(dataset, proportion):
    datacut = dataset.cut(proportion)
    x_test = datacut.get_RHS()
    y_test = datacut.get_solutions()
    x_train = dataset.get_RHS()
    y_train = dataset.get_solutions()
    dataset.merge(datacut)
    return x_train, y_train, x_test, y_test


class optimiser:

    # data : class dataset
    # nn : class nn_hyperas
    # trials : objet of calss hyperopt
    def __init__(self, dataset, proportion, max_evals = 100, model=None):
        self.model = model,
        self.data = get_data(dataset, proportion)
        self.algo = tpe.suggest
        self.max_evals = max_evals
        self.trials = Trials()

    def get_data(self):
        return self.data

    def get_x_test(self):
        return self.data[2]

    def get_y_test(self):
        return self.data[3]

    def get_x_train(self):
        return self.data[0]

    def get_y_train(self):
        return self.data[1]

    def optimise_nn(self):
        x_train, y_train, x_test, y_test = self.data
        best_run, best_model = optim.minimize(model=self.model,
                                              data=self.data,
                                              algo=self.algo,
                                              max_evals=self.max_evals,
                                              trials=self.trials)
        print("Evalutation of best performing model:")
        print(best_model.evaluate(x_test, y_test))
        print(best_run)

class nn_hyperas:

    def __init__(self,x_train, y_train, x_test, y_test, batch_choice):
        self.model = Sequential()
        self.loss = "mean_squared_error"
        self.optimizer = "sgd"
        self.metrics = ["mean_squared_error"]
        self.is_compiled = False
        self.file_name = None
        self.rms = RMSprop()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.batch_choice = batch_choice

    def basic_nn(self, list_n_neurons):
        assert isinstance(list_n_neurons, list) or isinstance(list_n_neurons, list)
        self.__init__(self.x_train,self.y_train,self.x_test,self.y_test,self.batch_choice)
        for nb_neurons in list_n_neurons:
            self.add_relu(nb_neurons)
        self.add_sigmoid(1)
        rms = self.rms

    def add_relu(self, nb_neurons):
        self.model.add(Dense({{choice([nb_neurons])}}))
        self.model.add(Activation('relu'))
        self.model.add(Dropout({{uniform(0, 1)}}))

    def add_sigmoid(self, nb_neurons):
        self.model.add(Dense(nb_neurons))
        self.model.add(Activation('sigmoid'))

    def compile(self):
        if not self.is_compiled:
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
            self.is_compiled = True

    def fit(self,nb_epoch = 10, verbose = 2):
        return self.model.fit(self.x_train, self.y_train, batch_size = {{choice(self.batch_choice)}}, nb_epoch = nb_epoch,
                       verbose = verbose, validation_data = (self.x_test, self.y_test))

    def evaluate(self):
        score, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test accuracy:', acc)
        return score, acc

    def set_optimizer(self, optimizer_name):
        self.optimizer = optimizer_name

    def set_loss(self, loss_name):
        self.loss = loss_name

    def set_metrics(self, metrics_name):
        self.metrics = [metrics_name]

    def get_validation_acc(self):
        return np.amax(self.fit().history['val_acc'])

    def get_model(self):
        return {'loss': -self.get_validation_acc, 'status': STATUS_OK, 'model': self.model}

dataSet = dataset.dataset("petits_problemes_1-000")
trainAndTestData = get_data(dataSet,0.3)
myOptimiser = optimiser(dataset= dataSet, proportion=0.3,
                        model=nn_hyperas(trainAndTestData[0],trainAndTestData[1],trainAndTestData[2],trainAndTestData[3],batch_choice = [64, 128]).basic_nn([10,10]).get_model())


myOptimiser.optimise_nn()
