import cplex
import numpy as np

#class representing linear optimization problems. Will essentially be used to generate new linear optimization problems 
#by adding some noise (gaussian) to some chosen coefficients of a given problem. 

#Parameters: name, dev (stands for standard deviation, see comment of generate_random_prob),
#            prob (instance of the class cplex.Cplex, linear optimization problem), non_fixed_vars (subset of the variables of the
#            linear optimization problem, contains all variables we choose not to be fixed for all RTE-like problems and thus will be
#            affected by the noise we apply while the generation of random new problems)

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

#method solves the linear optimization problem

    def solve(self):
        self.prob.solve()



#This method generates a single random new problem by adding a gaussian noise
#to each variable of the RHS of the optimization problem.

#The standard deviation of that noise in each variable is computed by multiplying the
#value that variable takes by the factor dev, that can be set by the use of the method
#set_deviation. Thus the standard deviation is always chosen relative to the variable's
#value.

#Arguments taken: name of the new problem (type: string)
#Output: new problem (instance of the class problem)

    def generate_random_prob(self, name):
        list_rhs = self.prob.linear_constraints.get_rhs()
        l = len(list_rhs)
        for elem in non_fixed_vars:
            val = list_rhs[elem]
            list_rhs[elem] = val + np.random.normal(0, val*self.dev, 1)  # add gaussian noise to the RHS
        new_list = []
        for i in range(l):
            new_list.append((i, list_rhs[i]))
        new_prob = cplex.Cplex()  # create new problem
        new_prob.read(name)
        new_prob.linear_constraints.set_rhs(new_list)  # set the RHS to the bias RHS
        new_problem = RTElike_lin_opt_problem(new_prob)  # generate a new instance of the class
        return new_problem



# This method generates a given number of random new problems by adding gaussian noises
# to the instance of the class the method is applied to.

# Arguments taken: number of problems to be generated (type: int)
# Output: list of the generated problems (list of instances of the class problem)

    def generate_random_probs(self, int):
        new_list = []
        for i in range(int):
            new_list.append(self.generate_random_prob(self, self.name + "new_(d%)" % int))
        return(new_list)


#Test with dev = 0.1

test = RTElike_lin_opt_problem('petit_probleme.lp')
test.set_deviation(0.1)  # set the delta of the gaussian noise
test.solve()
test.set_non_fixed_vars([1,2,4])
print(test.prob.solution.get_objective_value())

new_prob = test.generate_random_prob('new_petit_probleme.lp')  # generate a new problem from test
new_prob.solve()
print(new_prob.prob.solution.get_objective_value())
