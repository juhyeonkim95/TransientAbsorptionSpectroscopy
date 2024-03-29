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

    vs1 = []
    vs1_n = []
    vs2_n = []
    
    error_distributions = []

    if False:    
        data = (pump_ons - pump_offs)
        data = np.mean(data[:,:,20:], axis=1)

        cov = np.cov(data.T)
        plt.imshow(cov)
        plt.colorbar()
        plt.savefig("../real_data_plot/covariance.png", dpi=600)
        plt.savefig("../real_data_plot/covariance.svg", dpi=600)
        exit(0)

    # plt.show()
    #print(cov.shape)
    # exit(0)


    do_per_time_plot = False
    if do_per_time_plot:
        ys = np.mean(data[80:90, :], axis=0)
        st = 10
        et = 80
        inc = 20
        target_t = st
        fig = plt.figure(figsize=(8, 4))
        for i in range(len(eo_delays)):
            t = eo_delays[i]
            if t == target_t and target_t <= et:
                target_t += inc
                plt.plot(data[:, i], label=t)
        plt.axhline(y = 0.0, color='black', linestyle = '-')
        plt.savefig("../real_data_plot/per_time_plot.png", dpi=600)
        plt.savefig("../real_data_plot/per_time_plot.svg", dpi=600)
        exit(0)

        #plt.legend()
        #plt.show()

    #plt.plot(eo_delays, data[30])
    #plt.xlim(0,100)
    # print(eo_delays[5])
    # plt.imshow(data, cmap="rainbow", extent=[eo_delays[10], 1000, 0, 1])
    # plt.imshow(data, cmap="rainbow")
    # x_labels = eo_delays[::10]
    #plt.xticks(np.arange(0, len(eo_delays), 1), eo_delays)
    # plt.colorbar()
    # plt.show()
    # plt.tight_layout()
    #plt.xticks(eo_delays)
    # plt.xscale('log')
    # plt.savefig("../real_data_plot/2d_profile.png", dpi=600)
    # plt.savefig("../real_data_plot/2d_profile.svg", dpi=600)

    # plt.clf()
    # xs = eo_delays#[20:90]
    #ys = np.mean(data[80:90, :], axis=0)#[20:90]
    #fig = plt.figure(figsize=(8, 4))
    #plt.scatter(xs,ys)
    # plt.plot(xs,ys)
    # plt.xscale('log')
    #plt.yscale('symlog')
    #plt.savefig("../real_data_plot/sample_overall_shape.png", dpi=600)
    #plt.savefig("../real_data_plot/sample_overall_shape.svg", dpi=600)

    # exit(0)
    # print(eo_delays2)
    
    absorbance_diff_all = pump_ons - pump_offs
    absorbance_diff_avged_all = np.mean(absorbance_diff_all[:, m[0]:m[1], :], axis=(0,1))

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

        for k in trange(1):
            # start_time = 5 + 3 * (np.random.random() * 2 - 1)
            # end_time = 100 + 20 * (np.random.random() * 2 - 1)
            start_time = 0.5
            end_time = 80.0
            
            valid_ti_idxs = np.argwhere(np.logical_and(eo_delays > start_time, eo_delays < end_time))
            valid_ti_range = [np.min(valid_ti_idxs), np.max(valid_ti_idxs)]
            print(valid_ti_range)


            xs = eo_delays[valid_ti_range[0]:valid_ti_range[1]]
            ys = absorbance_diff_avged[valid_ti_range[0]:valid_ti_range[1]]

            first_idxs = np.argwhere(np.logical_and(eo_delays > 0.05, eo_delays < 0.3))
            # print(len(first_idxs))
            A = np.mean(absorbance_diff_avged_all[first_idxs])

            B = absorbance_diff_avged_all[valid_ti_range[1]-5:valid_ti_range[1]+5]
            B = np.mean(B)

            def normalize(v):
                return (v-B) / (A-B)
            def denormalize(v):
                return v * (A-B) + B
            
            ys_n = normalize(ys)
            xs2 = eo_delays2[3:]
            ys2 = absorbance_diff_avged2[3:]
            random_idx = np.random.permutation(len(xs2))
            xs2 = xs2[random_idx[0:len(xs)]]
            ys2 = ys2[random_idx[0:len(xs)]]

            ys2_n = normalize(ys2)

            ys2_n_all = normalize(absorbance_diff_avged2[3:])
            print(ys2_n_all.shape)
            error_distributions.append(ys2_n_all - np.mean(ys2_n_all))


            # (1) without normalize
            k = -1 if A < 0 else 1
            exp_function1 = lambda t, a, b, c: (a * np.exp(-t / (20 * b)) + c)
            v1, pcov1 = scipy.optimize.curve_fit(exp_function1,  xs,  ys, p0=[A, 1.0, B])

            # (2) with normalize
            exp_function2 = lambda t, b: np.exp(-(t) / (20 * b))
            v1_n, pcov1_n =scipy.optimize.curve_fit(exp_function2,  xs,  ys_n)

            # (3) ours
            v2_n, pcov2_n =scipy.optimize.curve_fit(exp_function2,  xs2,  ys2_n)

            # plot?
            plt.scatter(xs, normalize(ys))
            plt.scatter(xs2, normalize(ys2))
            
            new_xs = np.linspace(np.min(xs), np.max(xs), 2000)
            new_ys1 = exp_function1(new_xs, *v1)
            new_ys1_n = exp_function2(new_xs, *v1_n)
            new_ys2_n = exp_function2(new_xs, *v2_n)
            
            plt.plot(new_xs, normalize(new_ys1), color="C0", linestyle="--")
            plt.plot(new_xs, new_ys1_n, color="C0")
            plt.plot(new_xs, new_ys2_n, color="C1")

            # print(v1[1])
            # print(v1_n[0])
            # print(v2_n[0])

            vs1.append(v1[1])
            vs1_n.append(v1_n[0])
            vs2_n.append(v2_n[0])
            # plt.savefig("../real_data_plot/%d.png" % i, dpi=600)
            # plt.savefig("../real_data_plot/%d.svg" % i, dpi=600)
            # plt.clf()
            # plt.show()

        # plt.show()
    
    if True:
        error_distributions = np.asarray(error_distributions)
        error_distributions = error_distributions.flatten()
        plt.clf()
        (mu, sigma) = norm.fit(error_distributions)
        n, bins, patches = plt.hist(error_distributions, bins=50, density=True)
        
        y = norm.pdf(bins, mu, sigma)
        print(mu, sigma)
        l = plt.plot(bins, y, 'r--', linewidth=2)
        
        # # plt.hist(error_distributions.flatten(), bins=100)
        plt.xlim(xmin=-0.6, xmax=0.6)
        # plt.show()
        # plt.savefig("../real_data_plot/error_profile.png", dpi=600)
        # plt.savefig("../real_data_plot/error_profile.svg", dpi=600)
        
        exit(0)

    vs1 = np.asarray(vs1) * 20
    vs1_n = np.asarray(vs1_n) * 20
    vs2_n = np.asarray(vs2_n) * 20
    
    print(np.mean(vs1), np.std(vs1))
    print(np.mean(vs1_n), np.std(vs1_n))
    print(np.mean(vs2_n), np.std(vs2_n))
    
    # return



if __name__ == "__main__":
    expnames = [
        "4CzIPNindcm_800nJ_425nm_300um_2kHz",
        "4CzIPNindcm_800nJ__neartautime_425nm_300um_2kHz"
        ]
    
    read_and_process_data(expnames[0], expnames[1])

    # for expname in expnames:
    # process_all_and_export(expname)