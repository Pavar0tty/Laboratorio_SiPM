import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 

def graph(X, Y, A, B, Z, S, K):
    fig, ax = plt.subplots(nrows= 1, ncols=1)
    #plt.plot(X, A, label = "V_b = 54V")
    #ax.scatter(X, A)

    plt.plot(X, B, label = "V_b = 53V")
    ax.scatter(X, B)
#    plt.errorbar(X, B, 0.008069216556564512)

    plt.plot(X, Y, label = "V_b = 54V")
    ax.scatter(X, Y)
 #   plt.errorbar(X, Y, 0.008069216556564512)

    plt.plot(X, K, label = "V_b = 54.5V")
    ax.scatter(X, K)
  #  plt.errorbar(X, K, 0.008069216556564512)

    plt.plot(X, Z, label = "V_b = 55V")
    ax.scatter(X, Z)
   # plt.errorbar(X, Z, 0.008069216556564512)

    plt.plot(X, S, label = "V_b = 55.5V")
    ax.scatter(X, S)
   # plt.errorbar(X, S, 0.008069216556564512)

    plt.title('Staircase Plot')
    ax.set_yscale ("log")
    ax.set_xlabel("Threshold[mV]")
    ax.set_ylabel("Log(Frequency[kHz])")
    ax.legend()
    plt.show()

def main():
    X = np.linspace(0, 30, 30)
    Z = pd.read_csv("staircase_plot_bello.CSV")
    Y = Z["Th [mV] V_B 53_5V "]
    graph(Y, Z["F [kHz] V_B 53_5V"],Z["F [kHz] V_B 54V"],Z["F [kHz] V_B 54_5V"],Z["F [kHz] V_B 55V"],Z["F [kHz] V_B 55_5V"],Z["F [kHz] V_B 56V"])
main()


    