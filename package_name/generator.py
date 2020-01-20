# The purpose of this code is to provide a user-friendly function
# generating a dataset
# with N random new RHS from a list of chosen linear optimization problems
# and their N associated solutions.
# The name of this function is "problem_generator"
# For more information see its description at the end of the code


import cplex
import numpy as np
from package_name import dataset


# lin_opt_pbs is a class representing linear optimization problems.
# It will be used to generate new linear optimization problems
# by adding some noise (gaussian) to some chosen coefficients of a given problem.

# Parameters of an instance of lin_opt_pbs :
#           name_list : a list of string giving the name of the linear optimization problems
#           dev : a float setting the relative deviation of the variables when generating new problems
#           prob_list : a list of instances of the class cplex.Cplex
#           non_fixed_vars : a list containing all indices of the variables which will be affected by the noise
#               when generating new problems. If not given by the user, is calculated by the programm.


# To create a new instance of lin_opt_pbs one can give
# either a list of names (string) of files, each of which containing an instance of the class cplex.Cplex
# or a list of instances of the class cplex.Cplex


class lin_opt_pbs:
    def __init__(self, prob_name_list, non_fixed_vars=None):
        n = len(prob_name_list)
        if isinstance(prob_name_list[0], str):
            self.name_list = prob_name_list
            prob_list = [cplex.Cplex() for i in range(n)]
            for i in range(n):
                prob_list[i].read(prob_name_list[i])
            self.prob_list = prob_list
        if isinstance(prob_name_list[0], cplex.Cplex):
            name_list = []
            for i in range(n):
                name_list.append(("problem_(%d)", i))
            self.name_list = name_list
            self.prob_list = prob_name_list
        self.dev = 0
        self.non_fixed_vars = non_fixed_vars
        if non_fixed_vars is None:
            self.calculate_non_fixed_vars()

    def set_deviation(self, dev):
        self.dev = dev

    def get_deviation(self):
        return self.dev

    def set_non_fixed_vars(self, non_fixed_vars):
        self.non_fixed_vars = non_fixed_vars

    def get_non_fixed_vars(self):
        return self.non_fixed_vars

    def calculate_non_fixed_vars(self):
        n = len(self.name_list)
        list_rhs = []
        for i in range(n):
            list_rhs.append(self.prob_list[i].linear_constraints.get_rhs())
        nb_constraints = len(list_rhs[0])
        if n == 1:
            print("Attention, un seul problème a été fourni. Par défaut aucune variable fixée.")
            self.set_non_fixed_vars([i for i in range(nb_constraints)])
        constraints_max = []
        constraints_min = []
        for i in range(nb_constraints):
            max = list_rhs[0][i]
            min = list_rhs[0][i]
            for j in range(n):
                if list_rhs[j][i] > max:
                    max = list_rhs[j][i]
                if list_rhs[j][i] < min:
                    min = list_rhs[j][i]
            constraints_max.append(max)
            constraints_min.append(min)
        constraints_max = np.array(constraints_max)
        constraints_min = np.array(constraints_min)
        constraints_range = constraints_max - constraints_min
        non_fixed_vars = []
        for i in range(len(constraints_range)):
            if constraints_range[i] == 0:
                non_fixed_vars.append(i)
        self.set_non_fixed_vars(non_fixed_vars)

    # The method generate_random_prob generates a single random new problem
    # by adding a gaussian noise to each variable of a chosen RHS (= right hand side)
    # and adds it to self.prob_list

    # The standard deviation of that noise in each variable is computed by multiplying the
    # value that variable takes by the factor dev.
    # Thus the standard deviation is always chosen relative to the variable's value.

    # Arguments taken: an int giving the index of the chosen optimization problem
    # Output: None (the new problem has been added to self.prob_list)

    def generate_random_prob(self, k):
        list_rhs = self.prob_list[k].linear_constraints.get_rhs()
        l = len(list_rhs)
        for elem in self.non_fixed_vars:
            val = list_rhs[elem]
            list_rhs[elem] = val + (np.random.normal(0, abs(val) * self.dev, 1))[0]  # add gaussian noise to the RHS
        new_list = []
        for i in range(l):
            new_list.append((i, list_rhs[i]))
        new_prob = cplex.Cplex()  # create new problem
        new_prob.read(self.name_list[k])
        new_prob.linear_constraints.set_rhs(new_list)  # set the RHS to the bias RHS
        self.name_list.append(("problem_(%d)", len(self.prob_list)))
        self.prob_list.append(new_prob)

    # The method generate_random_prob_mult generates a given number N of random new problems
    # from an instance of lin_opt_pbs.

    # Arguments taken: number of problems to be generated (type: int)
    # Output: None (the new generated problems have been added to self.prob_list)

    def generate_random_prob_mult(self, N):  # Warning : random choice (by EtiMag)
        K = len(self.prob_list)
        for i in range(N):
            ind = np.random.randint(K)
            self.generate_random_prob(ind)

    # The method extract_RHS extracts some chosen coefficients from the RHS
    # of instances of lin_opt_pbs given in a list
    # and returns them in a list
    # The chosen coefficients are given by self.non_fixed_vars

    # Arguments taken: a lin_opt_pbs instance
    # Output: a list of truncated RHS (i.e. a list of list)

    def extract_RHS(self):
        new_list = []
        nb_pb = len(self.prob_list)
        for i in range(nb_pb):
            pb = self.prob_list[i]
            constraints = pb.linear_constraints.get_rhs()
            truncated_constraints = []
            for coeff in self.non_fixed_vars:
                truncated_constraints.append(constraints[coeff])
            new_list.append(truncated_constraints)
        return (new_list)

    # The method calculate_solutions determines the exact solutions
    # of the problems in an instance of lin_opt_pbs
    # and returns them in a list

    # Arguments taken: a lin_opt_pbs instance
    # Output: a list of solutions (float list)

    def calculate_solutions(self):
        new_list = []
        nb_pb = len(self.prob_list)
        for pb in range(nb_pb):
            (self.prob_list[pb]).solve()
            new_list.append((self.prob_list[pb]).solution.get_objective_value())
        return (new_list)


# The function problem_generator generates an instance of dataset
# with N random RHS based on a chosen linear optimization problem
# and their N associated solutions
# The RHS are truncated : only the non fixed coefficients are kept

# Parameters of problem generator :
#           problems : a string list giving the names of the linear optimization problems
#               OR a list of cplex.Cplex linear optimization problems
#           N : an int giving the number of RHS to generate
#           dev : a float setting the relative deviation of the variables when generating new problems
#           non_fixed_vars : a list containing all variables which will be affected by the noise
#               when generating new problems. If not given, calculated by the programm.
# Output:  a dataset instance containing N RHS and their N associated solutions

def problem_generator(problems, N, dev, non_fixed_vars=None):
    prob_root = lin_opt_pbs(problems, non_fixed_vars)
    prob_root.set_deviation(dev)
    prob_root.generate_random_prob_mult(N)
    rhs_list = prob_root.extract_RHS()
    sol_list = prob_root.calculate_solutions()
    data = dataset.dataset(rhs_list, sol_list)  # write either dataset or dataset.dataset to create a new instance
    return data


# The function small_problem_generator generates an instance of dataset
# with N random RHS with m non fixed coefficients chooed randomly based on a chosen linear optimization problem
# and their N associated solutions
# The RHS are truncated : only the non fixed coefficients are kept

# Parameters of problem generator :
#           problems : a string list giving the names of the linear optimization problems
#               OR a list of cplex.Cplex linear optimization problems
#           N : an int giving the number of RHS to generate
#           m : number of non fixed variables
#           dev : a float setting the relative deviation of the variables when generating new problems
# Output:  a dataset instance containing N RHS and their N associated solutions

def small_problem_generator(problem, N, m, dev):
    import random
    prob_root = lin_opt_pbs(problem)
    rhs= prob_root.prob_list[0].linear_constraints.get_rhs()
    non_fixed_rhs = random.sample(range(1,len(rhs)),m)
    prob_root.non_fixed_vars = non_fixed_rhs
    return problem_generator(problem, N, dev, non_fixed_rhs)


# Testing the methods defined above
# data = (problem_generator['petit_probleme.lp'], 10, 0.1, [23, 24, 25])
# print(data.get_RHS())
# print(data.get_solutions())
# data.dump_in_file("essai")
#
# new_dataset = dataset.dataset("essai")
# print("resultat")
# print(new_dataset.get_RHS())
