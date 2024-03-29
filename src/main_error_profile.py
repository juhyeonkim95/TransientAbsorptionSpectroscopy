import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
import os
import scipy
from utils.visualize_utils import *
from utils.loader_utils import *
from scipy.stats import norm
import matplotlib.mlab as mlab


def process_error():
    for i in range(1):
        # file_name = "Run 0 Cycle%d.csv" % (i+1)
        file_name = "Run 0 Avg Diff spec 01.csv"
        eo_delays, total_data  = load_long_data("no_sample", file_name=file_name)

        total_data = np.transpose(total_data, axes=[0, 2, 1])
        total_data = np.reshape(total_data, (-1, 256))
        
        v = np.mean(total_data, axis=0)
        # target_data = (total_data - v)
        target_data = total_data[:, 40:120]
        target_data = target_data.flatten()
        (mu, sigma) = norm.fit(target_data)

        n, bins, patches = plt.hist(target_data, bins=200, density=True)
        
        y = norm.pdf(bins, mu, sigma)
        print(mu, sigma)
        l = plt.plot(bins, y, 'r--', linewidth=2)

        plt.savefig("../real_data_plot/error_profile_background.png", dpi=600)
        plt.savefig("../real_data_plot/error_profile_background.svg", dpi=600)
        
        exit(0)


        # plt.show()
        
        # plt.xlabel('time')
        # plt.ylabel('wavelength')        
        # plt.show()


if __name__ == "__main__":
    process_error()