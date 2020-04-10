### A LIRE
# Ceci est la bibliothèque (=library en anglais) de nos fonctions.
#On cherche une fonction en utillisant ctrl+F et les mots clefs (=keywords en anglais)
#
# Règles : 
# 1) Ce fichier ne comporte que des définitions de fonctions et des commentaires
# 2) On met des mots clefs comme dans les exemples
# 3) On explique comme dans les exemples
# 4) On numérote comme dans les exemples (les numéros des exemples comptent)
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
#1
#exemple 1, nombres premiers, diviseurs, example
def est_premier(m):
    """ renvoie True si m est un entier >0 premier et False sinon """
    if isinstance(m,int) and m>0:
        for k in range(2,int(np.sqrt(m)) + 1):# On ne teste que les diiviseurs inférieurs ou égaux à la racine de m
            if m%k == 0:
                return False
    return True

#2
#exemple 2, suite de Fibonacci, fibonacci, example
def Fibo(n):
    """ vérifie que n est un entier positif et renvoie la n-ième valeur de la suite de Fibonacci. Le calcul n'est pas récursif et la complexité est en O(n). les indices soont pris à partir de 0. """
    assert isinstance(n, int) and n>=0
    l = [1,1]# l sera la liste des valeurs de la suite, jusqu'à la n-ième valeur
    for k in range(n-1):# On va ajouter les n-1 termes suivants de la suite à la liste. A la fin on aura donc les termes d'indices allant de 0 à n.
        l.append(l[-1] + l[-2])
    return l[n] # la liste est de longueur n+1 donc l[n] est le dernier terme, sauf lorsque n = 0.

#3
# recupere données, récupère données, get data, seconds membres, second member
def get_SecondMember(nom_fichier):
    '''renvoie le vecteur des seconds membres coontenus dans le fichier nom_fichier'''
    return np.vstack(np.array(pd.DataFrame(pd.read_pickle(nom_fichier))[0]))

#4
# récupère données, get data, solutions, Solutions
def get_Solutions(nom_fichier):
    '''renvoie le vecteur des soltions contenues dans le fichier nom_fichier'''
    return np.vstack(np.array(pd.DataFrame(pd.read_pickle(nom_fichier))[0]))

#5
# overfitting, affiche la perte, loss
def affiche_loss(history):
    """ affiche la courbe de la loss sur le training set et le validation set. Permt de contrôler si il y a un overfitting. history est le résultat de model.fit"""
    loss_curve = history.history["loss"]
    loss_val_curve = history.history["val_loss"]
    plt.plot(loss_curve, label = "Train")
    plt.plot(loss_val_curve, label = "Val")
    plt.legend(loc = 'upper left')
    plt.title("Loss")
    plt.show()
