import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
import os
import scipy
from utils.visualize_utils import *
from utils.loader_utils import *


def get_meaningful_range(expname):
    meaningful_range_dict = {
        "4CzIPN in acn": [60,70],
        "PC2NP in dcm": [130, 138],
        "PCNP in toluene": [121, 128], #[131, 136]
        "PCTNP in toluene": [106, 116]
    }
    return meaningful_range_dict.get(expname)

def get_meaningful_time_range(expname):
    meaningful_range_dict = {
        "4CzIPN in acn": [10, 80],
        "PC2NP in dcm": [15, 80],
        "PCNP in toluene": [10, 80], #[131, 136]
        "PCTNP in toluene": [10, 80]
    }
    return meaningful_range_dict.get(expname, [10, 90])



def lerp(a, b, x):
    return a * (1-x) + b * x




def integrate_signal(eo_delays, total_data):
    ys = total_data
    ys = np.mean(ys, axis=0)
    # ys -= np.mean(ys[-10:-1])

    ys /= np.max(np.abs(ys))
    xs = eo_delays
    
    max_idx = np.argwhere(np.abs(ys) == np.max(np.abs(ys)))[0][0]

    mt = get_meaningful_time_range(expname)
    xs = xs[mt[0]:mt[1]]
    ys = ys[mt[0]:mt[1]]

    K = xs.max()

    xs /= xs.max()

    plot_xs, plot_ys, old_ys = approximate_func(xs, ys)
    # plt.scatter(xs, ys-old_ys)

    plt.scatter(xs * K, ys)
    plt.plot(plot_xs * K, plot_ys)

    # xs_new, ys_new = get_new_sample(xs, ys, method="uniform", N=100)
    # plot_xs_new, plot_ys_new, old_ys = approximate_func(xs_new, ys_new)
    # plt.scatter(xs_new * K, ys_new)
    # plt.plot(plot_xs_new * K, plot_ys_new)

    # xs_new, ys_new = get_new_sample(xs, ys, method="proportional", N=100)
    # plot_xs_new, plot_ys_new, old_ys = approximate_func(xs_new, ys_new)
    # plt.scatter(xs_new * K, ys_new)
    # plt.plot(plot_xs_new * K, plot_ys_new)

    # plt.xscale('log')

    plt.show()

def get_new_sample(xs, ys, N=30, method="uniform", initial_guess=1.0, prop_power=-1.0):
    if method == "uniform":
        uniform_x = np.linspace(xs[0], xs[-1], N)
        f_uniform_x = np.interp(uniform_x, xs, ys)
        return uniform_x, f_uniform_x
    elif method == "proportional":
        exp_x = np.linspace(xs[0], xs[-1], N)
        fs = np.exp(-(exp_x-xs[0]) / initial_guess)
        fs = np.power(fs, prop_power)
        fs /= fs.sum()
        fs_cum = np.cumsum(fs) - fs[0]
        exp_x = lerp(xs[0], xs[-1], fs_cum)
        f_exp_x = np.interp(exp_x, xs, ys)
        return exp_x, f_exp_x

def approximate_func(xs, ys):
    k = 1 if ys[0] > 0 else -1

    exp_function = lambda t, a, b, c : k * (a * (np.exp(-t/b)) + c)


    v, _ =scipy.optimize.curve_fit(exp_function,  xs,  ys)
    new_xs = np.linspace(np.min(xs), np.max(xs), 200)
    new_ys = exp_function(new_xs, *v)
    old_ys = exp_function(xs, *v)
    
    print(*v)
    #A, tau, B = v
    #print(A, tau, B)

    return new_xs, new_ys, old_ys

def process_short(expname):
    eo_delays, total_data  = load_short_data(expname)

    output_folder = os.path.join("../plot/short", expname)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #plt.plot(eo_delays)
    #plt.show()
    
    # m = get_meaningful_range(expname)
    # m = []
    m = [20, 25]
    total_data = total_data[m[0]:m[1], :]
    #plt.plot(eo_delays, np.mean(total_data, axis=0))
    #plt.show()

    # print(total_data.shape)
    
    # integrate_signal(eo_delays, total_data)
    # print(len(eo_delays))
    # plt.plot(eo_delays)
    # plt.show()

    # plot_2d_graph(eo_delays, total_data)
    # return

    # plot_per_time_graph(eo_delays, total_data)
    integrate_signal(eo_delays, total_data)
    # return

    # print(eo_delays[0:12])
    # return

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot_surface(xx, yy, total_data, cmap=cm.coolwarm, linewidth=0)

    #plt3d = plt.figure().gca(projection='3d')
    #plt3d.plot_surface(xx, yy, total_data, alpha=0.2)
    #plt.show()

    #plt.imshow(total_datas)
    #plt.show()

    # print(np.isnan(np.sum(total_data)))
    # print(total_data.shape)
    # print(total_data.dtype)
    # return

    for j in trange(256):
        try:
            data = total_data[j, :]
            #data = df.iloc[j, :]
            #data = np.asarray(data).astype(float)

            xs = eo_delays[0:]
            ys = data[0:]
            plt.scatter(xs, ys)
            plt.savefig(os.path.join(output_folder, "%d.png" % j))
            plt.clf()
        
        except:
            continue

def process_long(expname):
    eo_delays, total_data  = load_long_data(expname)
    plt.imshow(total_data[0,:,:])
    plt.show()

    # repetition, wavelength, time
    
    m = get_meaningful_range(expname)

    #total_data = total_data[0,:,:] - total_data[-1,:,:]
    #plt.plot(np.mean(total_data[m[0]:m[1],:], axis=0))
    # plt.plot(np.mean(total_data[0,m[0]:m[1],10:80]-total_data[-1,m[0]:m[1],10:80], axis=1))
    #plt.show()
    total_data = total_data - total_data[-1, :, :]

    target_data = np.mean(total_data[10:,:,:], axis=1)

    target_data = target_data - np.mean(target_data, axis=0)
    target_data = target_data.flatten()
    plt.hist(target_data, bins=100)
    plt.show()
    
    # target_data = np.mean(total_data[5:,:,:], axis=0)

    # plt.plot(np.mean(total_data[1:,m[0]:m[1],:], axis=(1,2)))
    # plt.show()


    #target_data = np.mean(total_data[10:,m[0]:m[1],:], axis=1)

    #plt.plot(np.mean(total_data[1:,m[0]:m[1],:], axis=(1,2)))

    #plt.show()

    #plt.scatter(eo_delays, np.mean(total_data[:,m[0]:m[1],:]))

    covariance = np.cov(target_data)
    covariance /= np.max(covariance)
    # print(covariance.shape)

    plt.imshow(covariance)
    plt.colorbar()
    plt.show()


    # print(covariance.shape)




if __name__ == "__main__":
    expnames = [
        "4CzIPNindcm_800nJ_425nm_300um_2kHz"
        #"4CzIPN in acn", 
        #"PC2NP in dcm", 
        #"PCNP in toluene", 
        #"PCTNP in toluene", 
        # "ruthenium catalyst"
        ]
    for expname in expnames:
        process_short(expname)