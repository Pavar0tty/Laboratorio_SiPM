import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def retta (Y):
    K= np.linspace(0, len(Y), 10000)
    Z=[]
    for i in range (10000):
        z = np.mean(Y)
        Z.append(z)
    return(K, Z)


def graph(X):
    fig, ax = plt.subplots(nrows=1, ncols= 1)
    ax.scatter(np.linspace(0, len(X), len(X)), X, color = 'red')
    plt.plot(retta(X)[0],retta(X)[1], label = 'Mean')
    plt.title('Caratterizzazione della deviazione standard')
    ax.set_xlabel('Counts')
    ax.set_ylabel ('Frequncy[kHz]')
    ax.legend()
    plt.show()

def main():
    Z = [0.369167,0.355867,0.355467,0.3673,0.3667,0.346633,0.352]
    graph(Z)
    print(np.std(Z))
    print(np.mean(Z))
    print('Errore relativo:', (np.std(Z)/np.mean(Z))*100, '%')
main()