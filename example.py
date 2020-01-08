from package_name.NeuralNetwork import nn
from package_name.dataset import dataset, RHS, solutions
from package_name.generator import lin_opt_pbs, problem_generator
from package_name.analyse import to_analyze

### Creating the data
problem, N, dev, List = ["test_problem1.mps", "test_problem2.mps"], 10000, 0.1, [1,2,3,4,5,6] # Arguments for problem_generator
problem_set = problem_generator(problem, N, dev, non_fixed_vars= List) # class dataset
### Modifying the data before training
problem_set.RHS.normalize_standard()
problem_set.solutions.box() # Boxplot of the solutions
### Creating th Neural Network
layers_list = [100, 100, 10]
epochs = 10
neural_network = nn()
neural_network.basic_nn(layers_list) #neural_network is now an instance of nn with the layers [Dense(100, relu), Dense(100, relu), Dense(1,  sigmoid)]

### training the neural network
analyze_set = neural_network.train_with(problem_set, epochs, validation_split=0.0) #problem_set is an instance of dataset

### Analyzing the results
print("proportion of predictions over relative precision ", analyze_set.hoped_precision, " is ", analyze_set.rate_over_precision())
print("number of predictions over relative precision ", analyze_set.hoped_precision, " is ", analyze_set.size*analyze_set.rate_over_precision())
analyze_set.precision_histogram() #plots the relative precision histogram

