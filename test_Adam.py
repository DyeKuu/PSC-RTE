from package_name.dataset import dataset
from package_name.NeuralNetwork import nn
import matplotlib as plt

### Creating the data
# data = problem_generator(['petit_probleme.lp'], 100000, 0.1, [23, 24, 25])
problem_set_for_training = dataset("petits_problemes_N100-000_dev0.1")
problem_set_for_evaluation = problem_set_for_training.cut(0.2)

### Creating the Neural Network

layers_list, last_activation, epochs, neural_network = [4], None, 20, nn()
neural_network.basic_nn(layers_list, last_activation)
neural_network.set_loss("mean_absolute_percentage_error")
neural_network.set_metrics(["mean_absolute_percentage_error"])
neural_network.set_optimizer("Adam")
neural_network.add_processing_linear_mean()

### Training the neural network
training_data = neural_network.train_with(problem_set_for_training, epochs, 0.1)

### Evaluating the neural network
evaluation_data = neural_network.predict(problem_set_for_evaluation)
training_data.hoped_precision = 0.001
evaluation_data.hoped_precision = 0.001

#print(training_data.history.history)
#training_histogram = training_data.precision_histogram("For training dataset : ")
print(evaluation_data.mean_squared_error())
print(evaluation_data.mean_precision_error())
evaluation_histogram = evaluation_data.precision_histogram("For evaluation dataset : ")
