![](https://raw.github.com/DyeKuu/PSC-RTE/master/report/icon.png)
# Collective Scientific Project

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

### Group Member: Eulalie Creusé - Étienne Maeght - Yiming Qin - Arthur Schichl - Kunhao Zheng
### Tuteur : M. Manuel Ruiz
### Coordinateur : M. Emmanuel Haucourt

This is a project by 5 students of Ecole Polytechnique, cooperated with RTE who has provided the subject. The object of this project is to explore the possibility of solving a large number big linear programming problems, which vary only in the part of RHS (Right-Hand Side) in the constraints, by means of machine learning.

## Table of Contents

- [Strategy](#strategy)
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [API](#api)
- [Contributing](#contributing)
- [License](#license)

## Strategy

generator -> (generates) -> instance of dataset -> (is given to) -> instance of nn -> (generates) -> instance of to_analyse

data.py and NeuralNetwork.py contain basic classes and methods for both datasets and Neural Networks.

"exemple" links all files and classes together

"petits_problemes_1-000" is a file containing a dataset (1000 "petits problèmes") made from "petit_probleme.lp" with dev = 0.1



# PSC-RTE

### Three Steps to Use!
### 1. Download and dezip
### 2. Right click in folder : Open in Pycharm as Project
### 3. Run the file example.py

We put the "Bilan Réunion" and other discussions in the "Issues".

Site for hyperas https://github.com/maxpumperla/hyperas
