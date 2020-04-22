from package_name.NeuralNetwork import nn
from package_name.generator import problem_generator_y
from package_name.generator import calculate_network_size
import numpy as np
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#problem_file = 'petit_probleme.lp'
#network_size = [23, 100, 46]
problem_file = 'SP_1_50_230.lp'
network_size = calculate_network_size([problem_file])
print("network size", network_size)

### Creating the data

num_train = [100000]
num_epoque = [350]
batch_size = 100

error = np.zeros((len(num_train), len(num_epoque)))

for i in range(len(num_train)):

    for j in range(len(num_epoque)):

        ### Creating the Neural Network
        layers_list, last_activation, epochs, neural_network = network_size, None, num_epoque[j], nn()
        neural_network.basic_nn(layers_list, last_activation)
        neural_network.set_loss("mean_absolute_percentage_error")
        neural_network.set_metrics(["mean_absolute_percentage_error"])
        neural_network.set_optimizer("Adam")

        ### Setting the learning rate update

        def scheduler(epoch):
            if epoch < 5:
                return 0.1 / (epoch + 1)
            elif epoch < 10:
                return 0.01 / (2 * epoch - 9)
            elif epoch < 100:
                return 0.001 * np.exp(0.06 * (10 - epoch))
            else:
                return 0.0000025

        def schedulerI(epoch):
            if epoch < 5:
                return 0.1 / (epoch + 1)
            elif epoch < 10:
                return 0.01 / (2 * epoch - 9)
            elif epoch < 30:
                return 0.001
            elif epoch < 300:
                return 0.001 * np.exp(0.06 * (10 - epoch//3))
            else:
                return 0.0000025

        callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

        probgenerator = problem_generator_y([problem_file], batch_size, 0.1)
        validgenerator = problem_generator_y([problem_file], 20, 0.1)

        ### Training the neural network
        neural_network.train_with_generator(probgenerator, epochs, num_train[i]//batch_size, validgenerator, [callback])

        ### Evaluating the neural network
        problem_set_for_evaluation = problem_generator([problem_file], num_train[i], 0.1)
        evaluation_data = neural_network.predict(problem_set_for_evaluation)
        evaluation_data.hoped_precision = 0.001

        print(evaluation_data.mean_squared_error())
        error[i][j] = evaluation_data.mean_precision_error()
        print("result", evaluation_data.mean_precision_error())

print(error)