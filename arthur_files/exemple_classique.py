from package_name.dataset import dataset
from package_name.generator import problem_generator
from package_name.generator import calculate_network_size
from package_name.NeuralNetwork import nn
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

# problem_file = 'petit_probleme.lp'
problem_file = 'SP_1_50_230.lp'

### Creating the data

begin = time.time()
problem_set_for_training = problem_generator([problem_file], 500000, 0.1)
end = time.time()
time2 = (end - begin)/5

#problem_set_for_training = dataset("petits_problemes_N100-000_dev0.1")
problem_set_for_evaluation = problem_set_for_training.cut(0.2)

### Creating the Neural Network
layers_list, last_activation, epochs, neural_network = calculate_network_size([problem_file]), None, 120, nn()
neural_network.basic_nn(layers_list, last_activation)
neural_network.set_loss("mean_absolute_percentage_error")
neural_network.set_metrics(["mean_absolute_percentage_error"])
neural_network.set_optimizer("Adam")

#neural_network.add_processing_linear_mean()

### Setting the learning rate update
def scheduler(epoch):
#  if epoch > 20 and epoch % 10 == 0:
#    return 0.01
#  elif epoch > 20 and epoch % 10 == 1:
#    return 0.001
#  elif epoch > 20 and epoch % 10 == 2:
#    return 0.0001
  if epoch < 5:
    return 0.1/(epoch+1)
  elif epoch < 10:
    return 0.01/(2*epoch-9)
  elif epoch < 100:
    return 0.001 * np.exp(0.06 * (10 - epoch))
  else:
    return 0.0000025

def schedulerI(epoch):
  if epoch < 5:
    return 0.1/(epoch+1)
  elif epoch < 10:
    return 0.01/(2*epoch-9)
  elif epoch < 70:
    return 0.001 * np.exp(0.06 * (10 - epoch))
  else:
    return 0.001 * np.exp(0.06 * (10 - 70)) / (epoch - 69)

def schedulerII(epoch):
  return 0.1/(10*epoch+1)

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

### Training the neural network
training_data = neural_network.train_with(problem_set_for_training, epochs, 0.3, [callback])

### Evaluating the neural network

begin = time.time()
evaluation_data = neural_network.predict(problem_set_for_evaluation)
end = time.time()

time1 = end - begin

training_data.hoped_precision = 0.001
evaluation_data.hoped_precision = 0.001

#print(training_data.history.history)
#training_histogram = training_data.precision_histogram("For training dataset : ")
evaluation_histogram = evaluation_data.precision_histogram("For evaluation dataset : ")

loss_curve = training_data.history.history["loss"]
loss_val_curve = training_data.history.history["val_loss"]

print(evaluation_data.mean_squared_error())
print(evaluation_data.mean_precision_error())

# plt.plot(loss_curve, label = "Train")
# plt.plot(loss_val_curve, label = "Val")
# plt.legend(loc = 'upper left')
# plt.title("Loss")
# plt.show()

print(time1)
print(time2)
print(time1/time2)