from package_name.NeuralNetwork import nn
from package_name.generator import problem_generator

### Creating the data
problem, N, dev, List = ["petit_probleme.lp"], 6000, 0.0001, [23,24,25]
problem_set = problem_generator(problem, N, dev, non_fixed_vars=List)
problem_set2 = problem_set.cut(0.5)
### Creating the Neural Network
layers_list, epochs, neural_network = [20,20,20], 10, nn()
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