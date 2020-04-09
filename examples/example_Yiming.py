from package_name.regression_lineaire import RegressionLineaire
from package_name.generator import problem_generator_with_steady_modification_of_unique_constraint
from package_name.NeuralNetwork import nn
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import numpy as np

### Creating the data
problem, N, dev, List = ["petit_probleme.lp"], 6000, 0.1, [23]
problem_set = problem_generator_with_steady_modification_of_unique_constraint(problem, N, dev, non_fixed_var=List)
problem_set2 = problem_set.cut(0.5)


### Test the lineaire regression
l = RegressionLineaire(problem_set2)
l.set_step(0.000002)
l.set_nb_iteration(40000)
l.predict()
# l.sol_exact()
# l.rl_Internet()


### Test the neural network
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(3, activation="linear"))
# model.add(tf.keras.layers.Dense(1, activation="linear"))
# model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["mean_squared_error"])
# model.fit(x=problem_set.get_RHS(), y=problem_set.get_solutions(), epochs=10, validation_split=0.3)
# model.evaluate(problem_set2.get_RHS(), problem_set2.get_solutions())
# model.save("simple_nm.h5")

layers_list, last_activation, epochs, neural_network = [1], None, 60, nn()
neural_network.basic_nn(layers_list)
neural_network.metrics = ["mean_absolute_percentage_error"]
neural_network.loss = "mean_squared_error"
analyze_train = neural_network.train_with(problem_set, epochs, 0.3,1)
analyze_set = neural_network.predict(problem_set2)
analyze_set.hoped_precision = 0.0001
histogramme = analyze_set.precision_histogram()
errors = np.array(analyze_train.history.history["val_mean_absolute_percentage_error"][3:50])/100
errorlog = np.log10(errors)

## Visualisation of error in neural network
len = len(errors)
plt.plot(range(len), errorlog, 'r--', label='error')
plt.title('With neural network')
plt.xlabel('Iteration')
plt.ylabel('Log of mean absolute relative error')
plt.legend()
plt.show()
