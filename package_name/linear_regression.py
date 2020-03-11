import numpy as np
import math
import matplotlib.pyplot as plt
from package_name.dataset import dataset
from package_name.analyse import to_analyze

# This class is created to verify if we can solve linear problem by regression methode
class RegressionLineaire:
    def __init__(self, dataset):
        self.file_name = None
        self.step = 1
        self.reg = 0
        self.dataset = dataset
        self.data0 = self.dataset.get_RHS()
        self.data = np.hstack((np.ones((len(self.data0), 1)), self.data0))
        size_RHS = len(self.data0[0])
        self.theta = np.random.random(size_RHS + 1)
        self.nb_iteration = 1000
        self.min_prec = math.pow(10,-8)

    def set_step(self, step):
        self.step = step

    def set_reg(self, reg):
        self.reg = reg

    def set_nb_iteration(self, n):
        self.nb_iteration = n

    def set_prec(self, p):
        self.min_prec = p

    def fonction_lineaire(self, x):
        return self.theta.dot(x)

    # This functions uses gradient descent to predict the parameters of this linear problem
    def predict(self):
        sol = self.dataset.get_solutions()
        sol_inv = 1/sol
        # list_J = []
        list_err_rel = []
        print("begin")
        print(sol)

        for i in range(self.nb_iteration):
            x_theta = self.data.dot(self.theta)
            vect = sol - x_theta
            absvect = abs(vect)
            err_rel = 1/len(self.data)*sol_inv.dot(np.transpose(absvect))
            list_err_rel.append(err_rel)
            if ( err_rel < self.min_prec ):
                break
            delta_J = -2 * (np.transpose(self.data)).dot(np.transpose(vect))
            # step = max(1/abs(delta_J).max(),100)
            step = 2
            theta0 = self.theta - step * delta_J
            x_theta = self.data.dot(theta0)
            # print("sol: ", sol)
            # print("x_theta: ", x_theta)
            # print("delta_J: ", delta_J)
            # print("self.data: ", np.transpose(self.data))
            # print("vect: ", vect)
            vect = sol - x_theta
            J0 = 1 / len(self.data) * vect.dot(np.transpose(vect))

            for j in range(40):
                step = step/2
                theta = self.theta - step * delta_J
                x_theta = self.data.dot(theta)
                vect = sol - x_theta
                J = 1 / len(self.data) * vect.dot(np.transpose(vect))
                if J > J0:
                    print("J0: ", J0)
                    print("step: ", step)
                    break
                J0 = J
                theta0 = theta
            self.theta = theta0

        print("coef : ----------------------", self.theta)

        for i in range(len(list_err_rel)):
            list_err_rel[i] = math.log10(list_err_rel[i])
        plt.plot(range(len(list_err_rel)), list_err_rel, 'r--', label='error')
        plt.title('With gradient descent linear regression')
        plt.xlabel('Iteration')
        plt.ylabel('Log of mean absolute relative error')
        plt.legend()
        plt.show()

        object_to_analyze = to_analyze(sol, self.data.dot(self.theta))

        return object_to_analyze

    # This function returns the exact solution of the linear problem
    def sol_exact(self):
        sol = self.dataset.get_solutions()
        self.theta = np.linalg.inv((np.transpose(self.data).dot(self.data))).dot(np.transpose(self.data)).dot(sol)
        error = sol - self.data.dot(self.theta)
        print("coef : ----------------------", self.theta)
        print("error_systematique: ", error)
        print(type(error[1]))

        object_to_analyze = to_analyze(sol, self.data.dot(self.theta))

        return object_to_analyze

    # This function uses linear regression in package sklearn
    def rl_Internet(self):
        from sklearn.linear_model import LinearRegression
        regr = LinearRegression().fit(self.data0, self.dataset.get_solutions())
        print("score:", regr.score(self.data0, self.dataset.get_solutions()))
        print(self.data0.shape, self.dataset.get_solutions().shape)
        print(regr.coef_)
        # plt.scatter(self.data0, self.dataset.get_solutions(), color='black')
        # plt.plot(self.data0, regr.predict(self.data), color='red', linewidth=1)
        # plt.show()
