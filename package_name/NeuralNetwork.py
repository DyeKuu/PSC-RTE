import tensorflow as tf
from package_name.analyse import to_analyze
from package_name.dataset import dataset
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
# This class implements a neural network. The neural_network is trained and tested with an instance of dataset
# This class allows to modify the neural network

### example : how to create a basic neural network with a pre_processing
#
#   neural_network = nn()
#   neural_network.basic_nn([20, 20], last_activation = "sigmoid")     /!\ It is possible to set last_activation = None
#   neural_network.add_processing_linear_mean
#   neural_network.add_processing_add_const(1)
#
#   and neural_network is ready !



class nn:
    def __init__(self):
        self.model = tf.keras.models.Sequential()
        self.loss = "mean_squared_error"
        self.optimizer = "sgd"
        self.metrics = ["mean_squared_error"]
        self.is_compiled = False  # says if self.model already has been compiled with the layers, optmizer, loss and metrics
        self.file_name = None
        self.pre_processing = [] #contains the information about the pre treatment of the data. Post treatment is also defined by this field
        self.factor = 2.1 #only for linear pre treatment
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7,verbose=1, min_delta=1e-5, patience=2, cooldown=2, min_lr=0.000001)
        self.callback = [reduce_lr]

    def basic_nn(self, list_n_neurons, last_activation = None):
        """Initialises the network with layers whose numbers of neurons are given in the argument"""
        assert isinstance(list_n_neurons, list) or isinstance(list_n_neurons, list)
        self.__init__()
        for nb_neurons in list_n_neurons:
            self.add_relu(nb_neurons)
        assert last_activation == "relu" or last_activation == "sigmoid" or last_activation == None, "This activation name does not exist :D"
        if last_activation == None:
            self.add_no_activation(1)
        else:
            self.model.add(tf.keras.layers.Dense(1, activation = last_activation))

            
            
    def add_relu(self, nb_neurons):
        self.model.add(tf.keras.layers.Dense(nb_neurons, activation="relu"))

    def add_sigmoid(self, nb_neurons):
        self.model.add(tf.keras.layers.Dense(nb_neurons, activation="sigmoid"))
        
    def add_no_activation(self, nb_neurons):
        self.model.add(tf.keras.layers.Dense(nb_neurons))

    def compile(self):
        if not self.is_compiled:
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
            self.is_compiled = True

    def set_optimizer(self, optimizer_name):
        self.optimizer = optimizer_name

    def set_loss(self, loss_name):
        self.loss = loss_name

    def set_metrics(self, metrics_name):
        self.metrics = [metrics_name]
        
        

    def fit(self, pb_train, sol_train, epochs, validation_split, batch_size):
        return self.model.fit(x=pb_train, y=sol_train, epochs=epochs, validation_split=validation_split, batch_size= batch_size,callbacks = self.callback)
    
    
    
    def add_processing_linear_mean(self):
        """trains with a linear transform for the solutions so that mean is 0.5 and the values are between 0 and 1. Chosing factor > 2 makes you put the values tighter around 0.5 """
        self.pre_processing.append(["linear_mean", None, None]) # (name, a, b) for x -> a*x + b
        
    def add_processing_add_const(self, number_of_const_to_add):
        self.pre_processing.append(["add_const", number_of_const_to_add])
        
    def add_processing_linear_divby_max(self):
        self.pre_processing.append(["linear_max", None, None])
        
    def add_processing_standard(self):
        self.pre_processing.append(["linear_standard", None, None, 0, 0]) # contient ("linear_standard", list_sigma_coordinate_RHS, list_mean_coordinate_RHS, sigma_sol, mean_sol)
    
  

    def predict(self, dataset_instance):
        self.compile()
        new_dataset_instance = dataset_instance.copy()
        for processing in self.pre_processing: #pre_processing
            if processing[0] == "add_const":
                new_dataset_instance.RHS.add_const(processing[1])
            elif processing[0] == "linear_standard":
                rhs_matrix = new_dataset_instance.get_RHS()
                std = processing[1]
                mean = processing[2]
                rhs_matrix = (rhs_matrix - mean)/std
                new_dataset_instance.set_RHS(rhs_matrix)

        object_to_analyze = to_analyze(new_dataset_instance.get_solutions(), self.model.predict(new_dataset_instance.get_RHS()).flatten())

        for processing in self.pre_processing[::-1]: #post_processing
            if processing[0] == "linear_mean" or processing[0] == "linear_max":
                object_to_analyze.untransform_predictions_linear(processing[1], processing[2])
            elif processing[0] == "linear_standard":
                std = processing[3]
                mean = processing[4]
                object_to_analyze.untransform_predictions_linear(1/std, -mean/std)
        object_to_analyze.add_used_nn(self)
        return object_to_analyze
    

    def train_with(self, initial_dataset_instance, epochs, validation_split, batch_siz):
        """ Trains the network using the dataset. Arguments : class dataset Out : class to_analyze"""
        self.compile()
        assert isinstance(initial_dataset_instance, dataset)
        dataset_instance = initial_dataset_instance.copy()

        for processing in self.pre_processing: ### pre processing
            if processing[0] == "linear_mean":
                mean_value = np.mean(dataset_instance.get_solutions())
                max_abs = np.max(np.absolute(dataset_instance.get_solutions()-mean_value))
                a = 1/(self.factor*max_abs)
                b = - mean_value/(self.factor*max_abs) + 0.5
                processing[1], processing[2] = a,b # collecting the values a and b for pre processing x -> a*x + b
                dataset_instance.solutions.apply_linear(a, b)
            elif processing[0] == "add_const":
                dataset_instance.RHS.add_const(processing[1])
            elif processing[0] == "linear_max":
                max_sol = 1/np.max(abs(dataset_instance.get_solutions()))
                processing[1], processing[2] = max_sol, 0.0
                dataset_instance.solutions.apply_linear(processing[1], processing[2])
            elif processing[0] == "linear_standard":
                ##
                rhs_matrix = dataset_instance.get_RHS()
                std = np.std(rhs_matrix, axis=0)
                mean = np.mean(rhs_matrix, axis=0)
                print("std and mean shape ", std.shape, mean.shape)
                rhs_matrix = (rhs_matrix - mean) / std
                dataset_instance.set_RHS(rhs_matrix)
                processing[1] = std
                processing[2] = mean
                std_sol = np.std(dataset_instance.get_solutions())
                mean_sol = np.mean(dataset_instance.get_solutions())
                dataset_instance.solutions.apply_linear(1/std_sol, - mean_sol/std_sol)
                processing[3] = std_sol
                processing[4] = mean_sol
                ##

        history = self.fit(dataset_instance.get_RHS(), dataset_instance.get_solutions(), epochs=epochs, validation_split=validation_split, batch_size = batch_siz) # training the network
        object_to_analyze = self.predict(initial_dataset_instance)
        object_to_analyze.add_learning_history(history)
        object_to_analyze.add_used_nn(self)

        return object_to_analyze

    
    
    def save_model(self, name=None):
        """ Says the model with the given name. If no name is given, the previous name is used"""
        if name is None:  # If we don't give a name, then teh previous name is used
            assert self.file_name is not None, "No name :("
            name = self.file_name
        else:
            self.file_name = name  # if we give a name, then it is stored
        self.model.save(str(name) + ".h5")
