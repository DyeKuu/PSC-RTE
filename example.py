from package_name.NeuralNetwork import nn
from package_name.generator import problem_generator

### Creating the data
problem, N, dev, List = ["petit_probleme.lp"], 2000, 0.1, [23,24,25]
problem_set = problem_generator(problem, N, dev, non_fixed_vars=List)
### Modifying the data before training
problem_set.RHS.normalize_standard()
problem_set.solutions.normalize_standard()
### Creating th Neural Network
layers_list, epochs, neural_network = [10,10], 20, nn()
neural_network.basic_nn(layers_list)
### training the neural network
problem_set2 = problem_set.cut(0.5)
neural_network.train_with(problem_set, epochs, 0.3)
analyze_set = neural_network.predict(problem_set2)
analyze_set.hoped_precision = 0.5
### Analyzing the results
histogramme = analyze_set.precision_histogram()
histogramme[2][0].figure.savefig("exemple_RI.png")