#The objective of the class dataset is to manipulate RHSs and their associated solutions
#in order to make them exploitable by a neural network.


import numpy as np
import matplotlib.pyplot as plt
import pickle
from package_name.NeuralNetwork import nn

from sklearn.preprocessing import StandardScaler

class RHS:# contains a set of second members
    def __init__(self, data):#data can be a file name or an istance of np.array with the solutions
        if isinstance(data, np.matrix) or isinstance(data, np.array): #if it's a matrix
            self.content = data # then we get it
        if isinstance(data, list):
            self.content = np.array(data)
        else:
            raise Exception("Could not initialise the RHS instance. The data must be list or array.")

    def normalize_standard(self):
        scaler = StandardScaler()
        self.content = scaler.fit_transform(self.content)
    def max(self):
        """returns an array with the max value for each coordinate"""
        lmax = []
        (n,p) = self.content.shape
        for j in range(p):
            lmax.append(max(self.content[:,j]))
        return np.array(lmax)
    def min(self):
        """returns an array with the min value for each coordinate"""
        lmin = []
        (n,p) = self.content.shape
        for j in range(p):
            lmin.append(min(self.content[:,j]))
        return np.array(lmin)
    def range(self):
        """returns an array with the range for each coordinate"""
        return self.max() - self.min()
    def boxplot_range(self):
        """ boxplot of the range for each coordinate """
        plt.boxplot(self.range(), whis=[2.5,97.5])
    def size(self):
        """number of second members"""
        return self.content.size
    def get_RHS(self):
        return self.content

class solutions:
    def __init__(self, data):
        """data can be a list or an instance of np.array with the solutions"""
        if isinstance(data, np.array) or isinstance(data, list):# if data is a vector
            self.content = np.array(data)# then we get its value
        else:
            raise Exception("could not initialize the solution instance. The data must be list or array")
    def size(self):
        """number of solutions"""
        return self.content.size
    def mean(self):
        """mean value of the solutions"""
        return np.mean(self.content)
    def apply(self, f):
        """f is the function applied to every solution"""
        for i in range(self.size()):
            self.content[i] = f(self.content[i])
    def apply_linear(self, a, b):
        """ applies a linear transformation to every solution"""
        self.apply(lambda x: a*x + b)
    def box(self):
        """boxplot of the solutions"""
        plt.boxplot(self.content, whis=[2.5, 97.5])
    def get_solutions(self):
        """return an array with the solutions"""
        return self.content

class dataset:
    #Parameters of an instance of dataset :
#           RHS : an instance of RHS
#           solutions : an instance of solutions
# to modify dataset, use the methods from RHS and solutions
    
    def __init__(self, RHS_list, solutions_list = None):
        if isinstance(RHS_list, str) and solutions_list == None: # if RHS_list is a file name
            set = pickle.load(open(RHS_list, "rb")) #then we get the content of the file
            self.RHS = RHS(set[0])
            self.solutions = solutions(set[1])
        else: # if it is data, then we directly initialize the fields
            self.RHS = RHS(RHS_list) #class RHS
            self.solutions = solutions(solutions_list) # class solutions
    
    def get_solutions(self):
        """returns the solutions as an array"""
        return self.solutions.get_solutions()

    def get_RHS(self):
        """returns the solutions as an array"""
        return self.RHS.get_RHS()
    def dump_in_file(self, file_name): # puts the content in a pickle file (we get it back with __init__)
        import pickle
        set = (self.RHS.get_RHS(), self.solutions.get_solutions())
        pickle.dump(set, open("file_name", "wb"))
