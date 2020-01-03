#The objective of the class dataset is to manipulate RHSs and their associated solutions
#in order to make them exploitable by a neural network.


import numpy as np
from sklearn.preprocessing import StandardScaler


class dataset:
    
    
    #Parameters of an instance of dataset :
#           RHS : a list of RHS
#           solutions : a list of solutions
# solutions[i] is expected to be the solution of the linear optimisation problem with RHS[i]
    
    def init(self, RHS, solutions):
        self.RHS = RHS
        self.solutions = solutions
    
    def get_solutions(self):
        """returns the solutions"""
        return self.solutions
    
    def get_RHS(self):
        """returns the solutions"""
        return self.RHS
    
    def eliminate_under(self, upper_limit):
        """Eliminates the coordinates with range under the upper limit"""
        raise("function not defined yet :)")
    
    def transform_solution(self, f):
        """f is the function applied to every solution"""
        for i in range(self.size()):
            self.solutions[i] = f(self.solutions[i])
            
    def transform_solution_linear(self, a, b):
        """ applies a linear transformation to every solution"""
        for i in range(self.size()):
            self.solutions[i] = a*self.solutions[i] + b
    
    def normalize_standard_RHS(self):
        scaler = StandardScaler()
        self.RHS = scaler.fit_transform(self.RHS)
 

