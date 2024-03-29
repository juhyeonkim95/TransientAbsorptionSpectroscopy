import numpy as np
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import scipy
import pandas as pd

def run(method="uniform"):
    As = []
    Bs = []
    xs = []

    M = 10000
    for i in range(M):
        N = 1000
        if method == "uniform":
            x = np.linspace(0, 1, 1000)
        elif method == "proportional":
            x = np.linspace(0, 1, 1000)
            x = np.cumsum(x)
            x = x / np.sum(x)
        xs.append(x)
        y = x * 3 + np.random.normal(0, 0.1, size=1000) * x

        poly_A, poly_B = np.polyfit(x, y, 1)
        As.append(poly_A)
        Bs.append(poly_B)
    xs = np.mean(np.asarray(xs), axis=0)
    # plt.plot(xs)
    # plt.show()


    return As, Bs

    #plt.hist(As, bins=100)
    #plt.show()

def run_all():
    As1, Bs1 = run("uniform")
    As2, Bs2 = run("proportional")
    plt.hist(As1, bins=100)
    plt.hist(As2, bins=100)
    plt.show()
    
run_all()

# print(poly_A, poly_B)
