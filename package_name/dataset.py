# The objective of the class dataset is to manipulate RHSs and their associated solutions
# in order to make them exploitable by a neural network.


import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import StandardScaler


class RHS:  # contains a set of second members
    def __init__(self, data):  # data can be a file name or an istance of np.array with the solutions
        if isinstance(data, np.matrix) or isinstance(data, np.ndarray):  # if it's a matrix
            self.content = data  # then we get it
        elif isinstance(data, list):
            self.content = np.array(data)
        else:
            raise Exception("Could not initialise the RHS instance. The data must be list or array.")

    def normalize_standard(self):
        scaler = StandardScaler()
        self.content = scaler.fit_transform(self.content)

    def max(self):
        """returns an array with the max value for each coordinate"""
        lmax = []
        (n, p) = self.content.shape
        for j in range(p):
            lmax.append(max(self.content[:, j]))
        return np.array(lmax)

    def min(self):
        """returns an array with the min value for each coordinate"""
        lmin = []
        (n, p) = self.content.shape
        for j in range(p):
            lmin.append(min(self.content[:, j]))
        return np.array(lmin)

    def range(self):
        """returns an array with the range for each coordinate"""
        return self.max() - self.min()

    def boxplot_range(self):
        """ boxplot of the range for each coordinate """
        plt.boxplot(self.range(), whis=[2.5, 97.5])

    def size(self):
        """number of second members"""
        return len(self.content)

    def get_RHS(self):
        return self.content
    def add_const(self, number_of_const):
        n = self.size()
        const_matrix = np.ones((n, number_of_const))
        self.__init__(np.hstack((self.get_RHS(), const_matrix)))


class solutions:
    def __init__(self, data):
        """data can be a list or an instance of np.array with the solutions"""
        if isinstance(data, np.ndarray) or isinstance(data, list):  # if data is a vector
            self.content = np.array(data)  # then we get its value
        else:
            raise Exception("could not initialize the solution instance. The data must be list or array")

    def size(self):
        """number of solutions"""
        return len(self.content)

    def mean(self):
        """mean value of the solutions"""
        return np.mean(self.content)

    def normalize_standard(self):
        scaler = StandardScaler()
        self.content = scaler.fit_transform(self.content.reshape(-1, 1))

    def toSigmoid(self):
        from scipy.stats import logistic
        self.content= logistic.cdf(self.content)

    def apply(self, f):
        """f is the function applied to every solution"""
        for i in range(self.size()):
            self.content[i] = f(self.content[i])

    def apply_linear(self, a, b):
        """ applies a linear transformation to every solution"""
        self.apply(lambda x: a * x + b)

    def box(self):
        """boxplot of the solutions"""
        plt.boxplot(self.content, whis=[2.5, 97.5])

    def get_solutions(self):
        """return an array with the solutions"""
        return self.content


class dataset:
    # Parameters of an instance of dataset :
    #           RHS : an instance of RHS
    #           solutions : an instance of solutions
    # to modify dataset, use the methods from RHS and solutions

    def __init__(self, RHS_list, solutions_list=None):
        if isinstance(RHS_list, str) and solutions_list is None:  # if RHS_list is a file name
            set = pickle.load(open(RHS_list, "rb"))  # then we get the content of the file
            self.RHS = RHS(set[0])
            self.solutions = solutions(set[1])
        else:  # if it is data, then we directly initialize the fields
            self.RHS = RHS(RHS_list)  # class RHS
            self.solutions = solutions(solutions_list)  # class solutions
        s1, s2 = self.solutions.size(), self.RHS.size()
        assert s1 == s2, "RHS and solutions do not have the same size"

    def get_solutions(self):
        """returns the solutions as an array"""
        return self.solutions.get_solutions()

    def size(self):
        return self.RHS.size()

    def get_RHS(self):
        """returns the solutions as an array"""
        return self.RHS.get_RHS()

    def dump_in_file(self, file_name):  # puts the content in a pickle file (we get it back with __init__)
        import pickle
        set = (self.RHS.get_RHS(), self.solutions.get_solutions())
        pickle.dump(set, open(file_name, "wb"))

    def cut(self, proportion_to_cut):
        """cuts a random part of the dataset and returns a new dataset. The cut data is deleted from the first dataset"""
        size = self.size()
        number_to_cut = int(proportion_to_cut * size)
        index_to_cut = np.random.choice(size, number_to_cut, replace=False)  # We randomly generate the indexes to cut
        list_to_cut_bool = size * [False]
        for index in index_to_cut:
            list_to_cut_bool[index] = True  # list_to_cut_bool[i] is True if line i must be cut
        RHS_to_keep, solutions_to_keep = [], []
        RHS_to_cut, solutions_to_cut = [], []
        initial_RHS_array = self.get_RHS()
        initial_solutions_array = self.get_solutions()
        for i in range(size):
            if list_to_cut_bool[i]:  # if we cut the line i
                RHS_to_cut.append(initial_RHS_array[i])
                solutions_to_cut.append(initial_solutions_array[i])
            else:
                RHS_to_keep.append(initial_RHS_array[i])
                solutions_to_keep.append(initial_solutions_array[i])
        self.__init__(RHS_to_keep, solutions_to_keep)
        return dataset(RHS_to_cut, solutions_to_cut)

    def merge(self, other_dataset):
        """Merges the second dataset in the fist dataset. The second dataset is not modified."""
        assert isinstance(other_dataset, dataset), "It must be an instance of dataset"
        assert len(other_dataset.get_RHS()[0]) == len(self.get_RHS()[0]), "Coordinates do not have the same size :'("
        new_RHS_array = np.concatenate((self.get_RHS(), other_dataset.get_RHS()), axis=0)
        new_solutions_array = np.concatenate((self.get_solutions(), other_dataset.get_solutions()), axis=0)
        self.__init__(new_RHS_array, new_solutions_array)

    def copy(self):
        return dataset(np.copy(self.get_RHS()), np.copy(self.get_solutions()))

    def sol_fct_of_RHS(self):
        plt.plot(self.get_RHS(), self.get_solutions())
        plt.show()
        
    def set_similar(self, size = None):
    """sets all the RHS and solutions to make them similar to the first one"""
        if size == None:
            size = self.size()
        first_RHS = self.get_RHS()[0]
        first_solution = self.get_solutions()[0]
        RHS_list = []
        solutions_list = []
        for k in range(size):
            RHS_list.append(first_RHS.copy())
            solutions_list.append(first_solution)
        self.__init__(RHS_list, solutions_list)
        
    def cut_the_first_one(self):
        assert self.size()>0
        proportion_to_cut = 1/self.size()
        return self.cut(proportion_to_cut)
