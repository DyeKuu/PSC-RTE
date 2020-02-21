from package_name.dataset import dataset
from package_name.generator import problem_generator
from package_name.NeuralNetwork import nn
import matplotlib as plt

### Creating the data
# data = problem_generator(['petit_probleme.lp'], 100000, 0.1, [23, 24, 25])
problem_set_for_training = dataset("petits_problemes_N100-000_dev0.1")
problem_set_for_evaluation = problem_set_for_training.cut(0.2)

### Creating the Neural Network
layers_list, last_activation, epochs, neural_network = [100,10], None, 100, nn()
neural_network.basic_nn(layers_list, last_activation)
neural_network.set_loss("mean_absolute_percentage_error")
neural_network.set_metrics(["mean_absolute_percentage_error"])
#neural_network.add_processing_linear_mean()

### Training the neural network
training_data = neural_network.train_with(problem_set_for_training, epochs, 0.3)

### Evaluating the neural network
evaluation_data = evaluation_data = neural_network.predict(problem_set_for_evaluation)
training_data.hoped_precision = 0.001
evaluation_data.hoped_precision = 0.001

#print(training_data.history.history)
#training_histogram = training_data.precision_histogram("For training dataset : ")
evaluation_histogram = evaluation_data.precision_histogram("For evaluation dataset : ")

loss_curve = training_data.history.history["loss"]
loss_val_curve = training_data.history.history["val_loss"]

plt.plot(loss_curve, label = "Train")
plt.plot(loss_val_curve, label = "Val")
plt.legend(loc = 'upper left')
plt.title("Loss")
plt.show()