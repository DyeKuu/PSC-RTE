![](https://raw.github.com/DyeKuu/PSC-RTE/master/report/icon.png)
# Collective Scientific Project

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

### Group Member: Eulalie Creusé - Étienne Maeght - Yiming Qin - Arthur Schichl - Kunhao Zheng
### Tuteur : M. Manuel Ruiz
### Coordinateur : M. Emmanuel Haucourt

This is a project by 5 students of Ecole Polytechnique, cooperating with RTE who has provided the subject. The object of this project is to explore the possibility of solving a large number big linear programming problems, which vary only in the part of RHS (Right-Hand Side) in the constraints, by means of machine learning.

## Table of Contents

- [Strategy](#strategy)
- [Structure](#structure)
- [Contributing](#contributing)
- [License](#license)

## Strategy
![](https://raw.github.com/DyeKuu/PSC-RTE/master/report/strategy.png)


Given a small linear programming problem(in format .lp or .mps), we use the class [generator](https://raw.github.com/DyeKuu/PSC-RTE/master/package_name/generator.py) to add some gaussian noise to the coefficients thus to generate other similar problems.

Then by using the class [dataset](https://raw.github.com/DyeKuu/PSC-RTE/master/package_name/dataset.py), we encapsulate the RHS and its corresponding solution in it, forming the input and the label for the neural network.

We have also written a class [nn]() in [NeuralNetwork.py](https://raw.github.com/DyeKuu/PSC-RTE/master/package_name/NeuralNetwork.py) that encapsulates the neural network and all the methods related to it.

Finally, we pass all the data into a class [to_analyse](https://raw.github.com/DyeKuu/PSC-RTE/master/package_name/analyse.py) where we implement the codes for evaluation the performance and plot figures.

## Structure
### data
A folder where we store the data of the problem we generated from small problem.
### example
A folder which contains several example written by the group member to execute the code. It suffices to extract the file in the same directory as the folder [package_name](https://github.com/DyeKuu/PSC-RTE/tree/master/package_name).

### obsolete
A folder which contains mainly the old version of the code.
#### notebook_files
A folder where we find the files of jupyter notebook written mainly in the purpose of building the first neural network and using the package cplex to solve the linear programming problems.
### package_name
Our basic as well as important folder, where we have written the different class and skeleton for our project.
### report
A folder where you can find some figure for this README and our report.

## Contributing
Feel free to dive in! [Open an issue](https://github.com/DyeKuu/PSC-RTE/issues/new) or make a pull request.
### Contributors
This project exists thanks to all the people who contribute. 

[o-Eulalie-o](https://github.com/o-Eulalie-o)
[EtiMag](https://github.com/EtiMag)
[qym7](https://github.com/qym7)
[Kragon-Nox](https://github.com/Kragon-Nox)
[DyeKuu](https://github.com/DyeKuu)
## License

[GNU Affero General Public License v3.0](https://raw.githubusercontent.com/DyeKuu/PSC-RTE/master/LICENSE)
