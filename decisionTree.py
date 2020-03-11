import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor

from package_name.generator import problem_generator

# Data set
problem_set_for_training = problem_generator(['petit_probleme.lp'], 10000, 0.1, [25],path="")
problem_set_for_evaluation = problem_set_for_training.cut(0.2)

x = problem_set_for_training.get_RHS()
y = problem_set_for_training.get_solutions()

# Fit regression model
model2 = DecisionTreeRegressor(max_depth=20)
model3 = linear_model.LinearRegression()
model2.fit(x, y)
model3.fit(x, y)

# Predict
X_test = problem_set_for_evaluation.get_RHS()
y = problem_set_for_evaluation.get_solutions()
# print(X_test)
y_2 = model2.predict(X_test)
y_3 = model3.predict(X_test)

#create to anlalyze
from package_name.analyse import to_analyze
analyze_data = to_analyze(y,y_2)
hist = analyze_data.precision_histogram("")
plt.show()

#compare to linearRegression
linearReg_data = to_analyze(y,y_3)
hist = linearReg_data.precision_histogram("")
plt.show()