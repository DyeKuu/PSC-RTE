from package_name.NeuralNetwork import nn
from package_name.generator import problem_generator

### Creating the data
problem, N, dev, List = ["petit_probleme.lp"], 100, 1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Arguments for problem_generator
problem_set = problem_generator(problem, N, dev, non_fixed_vars=List)  # class dataset
### Modifying the data before training
problem_set.RHS.normalize_standard()
# problem_set.solutions.normalize_standard()
# problem_set.solutions.toSigmoid()
problem_set.solutions.box()  # Boxplot of the solutions
### Creating th Neural Network
layers_list = [10,10]
epochs = 10
neural_network = nn()
neural_network.basic_nn(layers_list)  # neural_network is now an instance of nn with the layers [Dense(100, relu), Dense(100, relu), Dense(1,  sigmoid)]

### training the neural network
analyze_set = neural_network.train_with(problem_set, epochs, 0.3)  # problem_set is an instance of dataset
print(problem_set.get_solutions())
print(neural_network.predict(problem_set))
print(problem_set.get_solutions()-neural_network.predict(problem_set))

### Analyzing the results
print("proportion of predictions over relative precision ", analyze_set.hoped_precision, " is ",
      analyze_set.rate_over_precision())
print("number of predictions over relative precision ", analyze_set.hoped_precision, " is ",
      analyze_set.size * analyze_set.rate_over_precision())
analyze_set.precision_histogram()  # plots the relative precision histogram
