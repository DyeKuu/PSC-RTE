from package_name.NeuralNetwork import nn
from package_name.generator import problem_generator_with_steady_modification_of_unique_constraint, problem_generator

### Creating the data
problem, N, dev, non_fixed_var = ["petit_probleme.lp"], 40, 0.0001, [23]
problem_set = problem_generator_with_steady_modification_of_unique_constraint(problem, N, dev, non_fixed_var)
# print(problem_set.get_solutions())
# print(problem_set.get_RHS())
problem_set2 = problem_set.cut(0.5)
### Creating the Neural Network
layers_list, epochs, neural_network = [], 10, nn()
neural_network.basic_nn(layers_list)
neural_network.metrics = ["mean_absolute_percentage_error"]
neural_network.loss = "mean_absolute_percentage_error"
neural_network.set_treatment_linear()
### Training the neural network
neural_network.train_with(problem_set, epochs, 0.3)
analyze_set = neural_network.predict(problem_set2)
analyze_set.hoped_precision = 0.0001

### Analyzing the results
histogramme = analyze_set.precision_histogram()
#histogramme[2][0].figure.savefig("exemple_RI.png")
