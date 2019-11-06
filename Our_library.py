### A LIRE
# Ceci est la bibliothèque (=library en anglais) de nos fonctions.
#On cherche une fonction en utillisant ctrl+F et les mots clefs (=keywords en anglais)
#
# Règles : 
# 1) Ce fichier ne comporte que des définitions de fonctions et des commentaires
# 2) On met des mots clefs comme dans les exemples
# 3) On explique comme dans les exemples
# 4) On numérote comme dans les exemples (les numéros des exemples comptent)

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