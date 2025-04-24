import pandas as pd
from iminuit import Minuit 
from iminuit.cost import LeastSquares
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2


def retta (x,a,b):
    return (a*x + b)


def read():
    X = []
    S = []
    Z = pd.read_csv('F_Vb.CSV')
    for i in range(len(Z.columns.values)):
        if i != 0:
            z = Z.iloc[:, i]
            x = float(np.mean(z))
            X.append(x)
            s = float(np.std(z))
            S.append(s)
            print(s)
    
    return (X, S)


def graph(X, S):
    Y = [52.5, 53, 53.5, 53.8, 54, 54.3, 54.5, 54.8, 55, 55.3, 55.5, 55.8, 56]

    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    plt.plot(Y, X)
    ax.scatter(Y, X)
    ax.errorbar(Y, X, S)
    plt.show()


def analisi(X, S):


    Y = [52.5, 53, 53.5, 53.8, 54, 54.3, 54.5, 54.8, 55, 55.3, 55.5, 55.8, 56]
    L_S = LeastSquares(X, Y, S, retta)
    my_minuit = Minuit (L_S, a = 0, b = 0)  
    my_minuit.migrad () 
    my_minuit.hesse ()   

#validità
    V = my_minuit.valid
    print('Validità: ', V)
    Q_squared = my_minuit.fval
    print ('Q-squared: ', Q_squared)
    N_dof = my_minuit.ndof
    print ('DOF: ', N_dof)
    p_v = 1 - chi2.cdf(Q_squared, N_dof)
    print('P value: ', p_v )

#valori dell'interpolazione
    a_f = my_minuit.values[0]
    b_f = my_minuit.values[1]

#visualizzazione 
    print(my_minuit)


def main():
    S = read()
    'graph(S[0], S[1])'
    analisi(S[0], S[1])



main()

