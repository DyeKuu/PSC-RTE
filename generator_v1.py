import cplex
import numpy as np


class problem:
    def __init__(self, prob_name):
        if isinstance(prob_name, str):
            self.dev = 0
            self.name = prob_name
            prob = cplex.Cplex()
            prob.read(prob_name)
            self.prob = prob
        if isinstance(prob_name, cplex.Cplex):
            self.dev = 0
            self.name = 'default'
            self.prob = prob_name

    def set_deviation(self, dev):
        self.dev = dev

    def get_deviation(self):
        return self.dev

    def solve(self):
        self.prob.solve()

    def generate_random_prob(self):
        list_rhs = self.prob.linear_constraints.get_rhs()
        list_rhs = list_rhs + np.random.normal(0, self.dev, len(list_rhs))  # add gaussian noise to the RHS
        new_list = []
        for i in range(len(list_rhs)):
            new_list.append((i, list_rhs[i]))
        new_prob = cplex.Cplex()  # create new problem
        new_prob.read(self.name)
        new_prob.linear_constraints.set_rhs(new_list)  # set the RHS to the bias RHS
        new_problem = problem(new_prob)  # generate a new class
        return new_problem


test = problem('petit_probleme.lp')
test.set_deviation(1)  # set the delta of the gaussian noise
test.solve()
print(test.prob.solution.get_objective_value())

new_prob = test.generate_random_prob()  # generate a new problem from test
new_prob.solve()
print(new_prob.prob.solution.get_objective_value())
