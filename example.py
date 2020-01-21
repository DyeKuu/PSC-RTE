from package_name.NeuralNetwork import nn
from package_name.generator import problem_generator

### Creating the data
problem, N, dev, List = ["petit_probleme.lp"], 100, 0.1, [23,24,25]  # Arguments for problem_generator
problem_set = problem_generator(problem, N, dev, non_fixed_vars=List)  # class dataset
### Modifying the data before training
problem_set.RHS.normalize_standard()
#problem_set.solutions.box()  # Boxplot of the solutions
### Creating th Neural Network
layers_list, epochs, neural_network = [10,10], 10, nn()
neural_network.basic_nn(layers_list)  # neural_network is now an instance of nn with the layers [Dense(100, relu), Dense(100, relu), Dense(1,  sigmoid)]

### training the neural network
problem_set2 = problem_set.cut(0.3) # cuts a random part of problem_set and returns a new dataset
neural_network.train_with(problem_set, epochs, 0.3)
analyze_set = neural_network.predict(problem_set2)

### Analyzing the results
print("proportion of predictions over relative precision ", analyze_set.hoped_precision, " is ",
      analyze_set.rate_over_precision())
analyze_set.precision_histogram()  # plots the relative precision histogram
