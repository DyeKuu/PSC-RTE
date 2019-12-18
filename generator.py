#The purpose of this code is to provide a user-friendly function
#generating N random new RHS from a chosen linear optimization problem
#and their N associated solutions
#The name of this function is "problem_generator"
#For more information see its description at the end of the code







import cplex
import numpy as np

#RTElike_lin_opt_problem is a class representing linear optimization problems.
#It will be used to generate new linear optimization problems 
#by adding some noise (gaussian) to some chosen coefficients of a given problem. 

#Parameters of an instance of RTElike_lin_opt_problem :
#           name : a string giving the name of the linear optimization problem
#           dev : a float setting the relative deviation of the variables when generating new problems
#           prob : an instance of the class cplex.Cplex, i.e. a linear optimization problem
#           non_fixed_vars : a list containing all variables which will be affected by the noise
#               when generating new problems.


#To create a new instance of RTElike_lin_opt_problem one can give
#either the name (string) of an instance of the class cplex.Cplex
#or an instance of the class cplex.Cplex

class RTElike_lin_opt_problem:
    def __init__(self, prob_name):
        if isinstance(prob_name, str):
            self.dev = 0
            self.name = prob_name
            prob = cplex.Cplex()
            prob.read(prob_name)
            self.prob = prob
            self.non_fixed_vars = []
        if isinstance(prob_name, cplex.Cplex):
            self.dev = 0
            self.name = 'default'
            self.prob = prob_name
            self.non_fixed_vars = []

    def set_deviation(self, dev):
        self.dev = dev

    def get_deviation(self):
        return self.dev
    
    def set_non_fixed_vars(self, non_fixed_vars):
        self.non_fixed_vars = non_fixed_vars
        
    def get_non_fixed_vars(self):
        return self.non_fixed_vars

    
#The method "solve" solves the linear optimization problem

    def solve(self):
        self.prob.solve()

#The method generate_random_prob generates a single random new problem
#by adding a gaussian noise to each variable of the RHS (= right hand side) of the optimization problem.

#The standard deviation of that noise in each variable is computed by multiplying the
#value that variable takes by the factor dev.
#Thus the standard deviation is always chosen relative to the variable's value.

#Arguments taken: none
#Output: new problem (instance of the class RTElike_lin_opt_problem)

    def generate_random_prob(self):
        list_rhs = self.prob.linear_constraints.get_rhs()
        l = len(list_rhs)
        for elem in self.non_fixed_vars:
            val = list_rhs[elem]
            list_rhs[elem] = val + (np.random.normal(0, abs(val)*self.dev, 1))[0]  # add gaussian noise to the RHS
        new_list = []
        for i in range(l):
            new_list.append((i, list_rhs[i]))
        new_prob = cplex.Cplex()  # create new problem
        new_prob.read(self.name)
        new_prob.linear_constraints.set_rhs(new_list)  # set the RHS to the bias RHS
        new_problem = RTElike_lin_opt_problem(new_prob)  # generate a new instance of the class
        return new_problem



#The method generate_random_prob_mult generates a given number N of random new problems
#by adding a gaussian noise to each variable of the RHS (= right hand side) of the optimization problem.

# Arguments taken: number of problems to be generated (type: int)
# Output: list of the generated problems (list of instances of the class problem)

    def generate_random_prob_mult(self, N):
        new_list = []
        for i in range(N):
            new_prob = self.generate_random_prob()
            new_prob.name = self.name + "new_(%d)", i
            new_list.append(new_prob)
        return(new_list)
 

#The method extract_RHS extracts some chosen coefficients from the RHS
#of instances of RTElike_lin_opt_problem given in a list
#and returns them in a list
#The chosen coefficients are given by self.non_fixed_vars

# Arguments taken: a list of RTElike_lin_opt_problem instances
# Output: a list of truncated RHS (i.e. a list of list)
        
    def extract_RHS(self, list_of_RTElike_lin_opt_problem):
        new_list = []
        nb_pb = len(list_of_RTElike_lin_opt_problem)
        for pb in range(nb_pb):
            constraints = (list_of_RTElike_lin_opt_problem[pb]).prob.linear_constraints.get_rhs()
            truncated_constraints = []
            for coeff in self.non_fixed_vars:
                truncated_constraints.append(constraints[coeff])
            new_list.append(truncated_constraints)
        return(new_list)
    
#The method calculate_solutions determines the exact solutions
#of instances of RTElike_lin_opt_problem given in a list
#and returns them in a list

# Arguments taken: a list of RTElike_lin_opt_problem instances
# Output: a list of solutions (float list)
    
    
    def calculate_solutions(self, list_of_RTElike_lin_opt_problem):
        new_list = []
        nb_pb = len(list_of_RTElike_lin_opt_problem)
        for pb in range(nb_pb):
            (list_of_RTElike_lin_opt_problem[pb]).prob.solve()
            new_list.append((list_of_RTElike_lin_opt_problem[pb]).prob.solution.get_objective_value())
        return(new_list)


#The function problem_generator generates N random RHS based on a chosen linear optimization problem
#and their N associated solutions
#The RHS are truncated : only the non fixed coefficients are kept

#Parameters of problem generator :
#           problem : a string giving the name of the linear optimization problem
#               OR a cplex.Cplex linear optimization problem
#           N : an int giving the number of RHS to generate
#           dev : a float setting the relative deviation of the variables when generating new problems
#           non_fixed_vars : a list containing all variables which will be affected by the noise
#               when generating new problems
# Output:  RHS_list : a list of truncated RHS
#          sol_list : a list of solutions

def problem_generator(problem, N, dev, non_fixed_vars):
    prob_root = RTElike_lin_opt_problem(problem)
    prob_root.set_deviation(dev)
    prob_root.set_non_fixed_vars(non_fixed_vars)
    prob_list = prob_root.generate_random_prob_mult(N)
    RHS_list = prob_root.extract_RHS(prob_list)
    sol_list = prob_root.calculate_solutions(prob_list)
    return (RHS_list, sol_list)
    


#Testing the methods defined above
#print(problem_generator('petit_probleme.lp', 5, 0.01, [23, 24, 25]))
