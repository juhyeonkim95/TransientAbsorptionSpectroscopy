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

def lerp(a, b, x):
    return a * (1-x) + b * x

def process_all_and_export(expname):
    N = 10
    pump_offs = []
    pump_ons = []
    eo_delays = []
    for i in range(10):
        eo_delay, pump_off, pump_on = load_unprocessed_data(expname, file_name="Run 0 Cycle%d.csv" % (i+1))
        pump_offs.append(pump_off)
        pump_ons.append(pump_on)
        eo_delays.append(eo_delay)
    
    pump_offs = np.asarray(pump_offs)
    pump_ons = np.asarray(pump_ons)
    eo_delays = np.asarray(eo_delays)

    output_path = os.path.join("../data/", expname)
    np.save(os.path.join(output_path, "pump_offs.npy"), pump_offs)
    np.save(os.path.join(output_path, "pump_ons.npy"), pump_ons)
    np.save(os.path.join(output_path, "eo_delays.npy"), eo_delays)
    return

def read_and_process_data(expname, expname2):

    output_path = os.path.join("../data/", expname)
    pump_offs = np.load(os.path.join(output_path, "pump_offs.npy"))
    pump_ons = np.load(os.path.join(output_path, "pump_ons.npy"))
    eo_delays = np.load(os.path.join(output_path, "eo_delays.npy"))
    
    output_path2 = os.path.join("../data/", expname2)
    pump_offs2 = np.load(os.path.join(output_path2, "pump_offs.npy"))
    pump_ons2 = np.load(os.path.join(output_path2, "pump_ons.npy"))
    eo_delays2 = np.load(os.path.join(output_path2, "eo_delays.npy"))

    eo_delays = eo_delays[0]
    eo_delays2 = eo_delays2[0]

    m = [80, 90]

    vs = []
    vs2 = []

    error_distributions = []
    # data = np.mean(pump_ons - pump_offs, axis=0)
    # plt.plot(eo_delays, data[30])
    # plt.xlim(0,100)
    # plt.imshow(data, cmap="rainbow")
    #plt.show()
    #exit(0)

    # print(eo_delays2)

    for i in range(10):
        absorbance_diff = pump_ons[i] - pump_offs[i]
        # absorbance_diff = pump_ons[i] - np.mean(pump_offs[i], axis=-1, keepdims=True)
        # absorbance_diff = np.mean(pump_ons - pump_offs, axis=0)
        # plt.scatter(eo_delays, pump_offs[i, 85])
        
        absorbance_diff_avged = np.mean(absorbance_diff[m[0]:m[1], :], axis=0)

        absorbance_diff2 = pump_ons2[i] - pump_offs2[i]
        # absorbance_diff2 = pump_ons2[i] - np.mean(pump_offs2[i], axis=-1, keepdims=True)
        # absorbance_diff2 = np.mean(pump_ons2 - pump_offs2, axis=0)
        absorbance_diff_avged2 = np.mean(absorbance_diff2[m[0]:m[1], :], axis=0)
        
        valid_ti_idxs = np.argwhere(np.logical_and(eo_delays > 4, eo_delays < 100))

        valid_ti_range = [np.min(valid_ti_idxs), np.max(valid_ti_idxs)]
        print(valid_ti_range)

        xs = eo_delays[valid_ti_range[0]:valid_ti_range[1]]
        ys = absorbance_diff_avged[valid_ti_range[0]:valid_ti_range[1]]
        
        B = absorbance_diff_avged[valid_ti_range[1]-2:valid_ti_range[1]+2]
        B = np.mean(B)
        # B = np.mean(absorbance_diff_avged[valid_ti_range[1]:])
        # print(B)

        first_idxs = np.argwhere(np.logical_and(eo_delays > 0.5, eo_delays < 1))
        print(np.min(first_idxs), np.max(first_idxs))
        # print(eo_delays[first_idxs])
        A = np.mean(absorbance_diff_avged[first_idxs])
        print(A)
        # print(first_idxs)
        # exit(0)
        # A = np.mean()
        # A = np.sort(np.abs(ys - B))[::-1]
        # A = np.mean(A[0:40])
        # A = np.max(np.abs(ys - B))
        ys = (ys - B) / (A - B)

        plt.scatter(xs, ys)

        
        xs2 = eo_delays2[3:]
        ys2 = absorbance_diff_avged2[3:]
        ys2 = (ys2 - B) / (A - B)

        random_idx = np.random.permutation(len(xs2))
        xs2 = xs2[random_idx[0:len(xs)]]
        ys2 = ys2[random_idx[0:len(xs)]]

        error_distributions.append(ys2 - np.mean(ys2))

        # plt.clf()
        # plt.hist(ys2 - np.mean(ys2))
        # plt.show()

        plt.scatter(xs2, ys2)

        exp_function = lambda t, b: np.exp(-(t-xs[0]) / (20 * b))

        v, pcov =scipy.optimize.curve_fit(exp_function,  xs,  ys)
        vs.append(v)
        print(v)
        print(np.sqrt(np.diag(pcov)))

        v2, pcov2 =scipy.optimize.curve_fit(exp_function,  xs2,  ys2)
        vs2.append(v2)
        print(v2)
        print(np.sqrt(np.diag(pcov2)))

        new_xs = np.linspace(np.min(xs), np.max(xs), 2000)
        new_ys = exp_function(new_xs, *v)
        new_ys2 = exp_function(new_xs, *v2)
        plt.plot(new_xs, new_ys)
        plt.plot(new_xs, new_ys2)

        # plt.show()
    
    # error_distributions = np.asarray(error_distributions)
    # error_distributions = error_distributions.flatten()

    # plt.clf()

    # (mu, sigma) = norm.fit(error_distributions)

    # n, bins, patches = plt.hist(error_distributions, bins=100, density=True)
    
    # y = norm.pdf(bins, mu, sigma)
    # print(mu, sigma)
    # l = plt.plot(bins, y, 'r--', linewidth=2)
    
    # # plt.hist(error_distributions.flatten(), bins=100)
    # plt.show()

    vs = np.asarray(vs)
    vs2 = np.asarray(vs2)
    
    print(np.mean(vs), np.var(vs))
    print(np.mean(vs2), np.var(vs2))
    
    # return



if __name__ == "__main__":
    expnames = [
        "4CzIPNindcm_800nJ_425nm_300um_2kHz",
        "4CzIPNindcm_800nJ__neartautime_425nm_300um_2kHz"
        ]
    
    read_and_process_data(expnames[0], expnames[1])

    # for expname in expnames:
    # process_all_and_export(expname)