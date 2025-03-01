import numpy as np
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import itertools
import sys


def exponential_with_three_params(t, A, tau, B):
    return A * np.exp(-t / tau) + B

def exponential_with_two_params(t, A, tau):
    return A * np.exp(-t / tau) + (1 - A)

def exponential_with_one_params(t, tau):
    return np.exp(-t / tau)

def exponential_with_one_params_derivative(t, tau):
    tau_prime = tau * (1.0 + 0.0 * (2.0 * np.random.random() - 1.0))
    return np.exp(-t / tau_prime) * t

def t_proportional(t):
    return t

def lerp(a, b, x):
    return a * (1-x) + b * x

def get_time_samples(**kwargs):
    time_sampling_method = kwargs.get("time_sampling_method", "proportional")
    st = kwargs.get("start_time", 0)
    et = kwargs.get("end_time", 4)
    N = kwargs.get("N", 100)

    if time_sampling_method == "uniform":
        ts = np.linspace(st, et, N)
    elif time_sampling_method == "proportional":
        ts = np.linspace(st, et, N)
        sample_f = kwargs.get("time_sampling_f")
        fs = sample_f(ts)
        fs /= fs.sum()
        fs_cum = np.cumsum(fs)
        ts = lerp(st, et, fs_cum)
    elif time_sampling_method == "proportional_sample":
        fixed_ts = np.linspace(st, et, 2000)
        sample_f = kwargs.get("time_sampling_f")
        fs = sample_f(fixed_ts)
        fs /= fs.sum()
        ts_idx = np.random.choice(len(fixed_ts), N, p=fs)
        ts = fixed_ts[ts_idx]
    elif time_sampling_method == "certain_range_only":
        ts = np.linspace(kwargs.get("range_start"), kwargs.get("range_end"), N)

    return ts

def get_sample_numbers(total_M, ts, **kwargs):
    sampling_number_method = kwargs.get("sampling_number_method", "uniform")
    N = len(ts)
    
    if sampling_number_method == "uniform":
        p = np.ones(N)
    elif sampling_number_method == "proportional":
        sample_f = kwargs.get("sample_number_f")
        fs = sample_f(ts)
        fs /= fs.sum()
        p = fs
    p = p.astype(float)
    p /= np.sum(p)
    Ms = total_M * p
    return Ms

def run_curve_fit(**kwargs):
    n_iter = kwargs.get("n_iter", 1000)
    N = kwargs.get("N", 100)
    parametric_model = kwargs.get("parametric_model")
    params = kwargs.get("params")

    fitted_params = []
    fitted_params_var = []

    noise_mode = kwargs.get("noise_mode", "normal")
    curve_fit_method = kwargs.get("curve_fit_method", "one")

    background_sample_ratio = kwargs.get("background_sample_ratio", -1.0)
    initial_sample_ratio = kwargs.get("initial_sample_ratio", -1.0)

    do_background_subtraction = background_sample_ratio > 0.0
    do_normalization = initial_sample_ratio > 0.0
    
    if curve_fit_method == "three":
        do_background_subtraction = False
        do_normalization = False
    elif curve_fit_method == "two":
        do_background_subtraction = True
        do_normalization = False
    elif curve_fit_method == "one":
        do_background_subtraction = True
        do_normalization = True
    elif curve_fit_method == "log_and_linear":
        do_background_subtraction = True

    force_noiseless = kwargs.get("force_noiseless", False)
    hetero = kwargs.get("hetero", False)
    hetero_f = kwargs.get("hetero_f", False)

    all_ts = []
    failed_count = 0

    # repetition
    for i in trange(n_iter):
        A = lerp(1.0, 2.0, np.random.random())
        B = lerp(0.0, 3.0, np.random.random())
        tau = lerp(1.0, 3.0, np.random.random())
        
        kwargs_new = {**kwargs}
        if "time_sampling_f" in kwargs:
            time_sampling_f = lambda t: kwargs.get("time_sampling_f")(t, tau)
            kwargs_new["time_sampling_f"] = time_sampling_f
        
        if kwargs.get("time_sampling_method") == "certain_range_only":
            kwargs_new["range_start"] = kwargs.get("range_start") * tau
            kwargs_new["range_end"] = kwargs.get("range_end") * tau

        # (1) set when to sample & how much to sample
        ts = get_time_samples(**kwargs_new)

        all_ts.append(np.sort(ts) / tau)
        
        # number of samples
        M = kwargs.get("M") 
        total_M = M * N

        # (1-1) first sample at t = 0? - for normalization
        if do_normalization:
            initial_sample_M = total_M * initial_sample_ratio #(initial_sample_ratio / (initial_sample_ratio + N + 1))
            total_M -= initial_sample_M
        
        # (1-2) last sample at t = infty? - for background subtraction
        if do_background_subtraction:
            background_sample_M = total_M * background_sample_ratio#(background_sample_ratio / (background_sample_ratio + N + 1))
            total_M -= background_sample_M

        Ms = get_sample_numbers(total_M, ts, **kwargs)

        # (2) Sample

        # single measurement stdev
        noise_stdev = kwargs.get("noise_stdev", 0.1)

        # exponential model
        observations_gt = A * np.exp(-ts/tau) + B
        error = np.random.normal(0, noise_stdev, size=len(ts)) * 1.0 / np.sqrt(Ms)
        if hetero:
            error *= hetero_f(observations_gt) # np.sqrt(observations_gt)

        observations = A * np.exp(-ts/tau) + B + error   

        # do background subtraction
        if do_background_subtraction:
            background_observation_gt = B
            error = np.random.normal(0, noise_stdev) / np.sqrt(background_sample_M)
            if hetero:
                error *= hetero_f(background_observation_gt)
            background_observation = background_observation_gt + error

            if force_noiseless:
                background_observation = B
            observations -= background_observation

        # do normalization
        if do_normalization:
            initial_observation_gt = A * np.exp(0.0) + B
            error = np.random.normal(0, noise_stdev) / np.sqrt(initial_sample_M)
            if hetero:
                error *= hetero_f(initial_observation_gt)
            initial_observation = initial_observation_gt + error
            if force_noiseless:
                initial_observation = A + B
            observations /= (initial_observation - background_observation)
        
        # (3) curve fitting
        try:
            if curve_fit_method == "three":
                f_fit = lambda t, A_hat, tau_hat, B_hat : A_hat * np.exp(-t/tau_hat) + B_hat
                params, pcov = scipy.optimize.curve_fit(f_fit, ts, observations)
                tau_expected = params[1]
                perr = np.sqrt(np.diag(pcov))
            
            elif curve_fit_method == "two":
                f_fit = lambda t, A_hat, tau_hat : A_hat * np.exp(-t/tau_hat)
                params, pcov = scipy.optimize.curve_fit(f_fit, ts, observations)
                tau_expected = params[1]
                perr = np.sqrt(np.diag(pcov))
            
            elif curve_fit_method == "one":
                f_fit = lambda t, tau_hat : np.exp(-t/tau_hat)
                params, pcov = scipy.optimize.curve_fit(f_fit, ts, observations)
                tau_expected = params[0]
                perr = np.sqrt(np.diag(pcov))
            
            elif curve_fit_method == "log_and_linear":
                valid_index = np.argwhere(observations > 1e-20).flatten()
                valid_observations = observations[valid_index]
                valid_ts = ts[valid_index]
                valid_observations = np.log(valid_observations)
                poly_A, poly_B = np.polyfit(valid_ts, valid_observations, 1)
                tau_expected = -1.0 / poly_A
                perr = 0.0

            fitted_params.append(((tau - tau_expected)/tau) ** 2)
            fitted_params_var.append(perr)
        except:
            failed_count += 1
            continue
    all_ts = np.asarray(all_ts)
    ts_avg = np.mean(all_ts, axis=0)
    print(failed_count)

    fitted_params = np.asarray(fitted_params)
    error = np.mean(fitted_params)
    error = np.sqrt(error)
    
    result = {}
    result["error"] = error
    result["ts"] = ts_avg
    result["observations"] = observations
    
    return result

def run_comparison_all(common_configs):
    total_errors = {}
    # tau = common_configs["params"]["tau"]

    expconfig_list = {
        "uniform": {
            "time_sampling_method": "uniform"
        },
        "exponential": {
            "time_sampling_method": "proportional_sample",
            "time_sampling_f": lambda t, tau: np.exp(t / -tau)
        },
        "exponential_derivative": {
            "time_sampling_method": "proportional_sample",
            "time_sampling_f": lambda t, tau: t * np.exp(t / -tau)
        },
        "exponential_derivative_power": {
            "time_sampling_method": "proportional_sample",
            "time_sampling_f": lambda t, tau: np.power(t * np.exp(t / -tau), 10)
        },
        "certain_range_only": {
            "time_sampling_method": "certain_range_only",
            "range_start": 0.99,
            "range_end": 1.01
        }
    }
    fig = plt.figure(figsize=(6,6))
    for expname, expconfig in expconfig_list.items():
        errors = {}

        # (1) Three model : Evaluate all of A, tau, B (without background subtraction & normalization)
        result = run_curve_fit(**common_configs, **expconfig, curve_fit_method="three")
        errors["no_normalize"] = result["error"]

        ts = result["ts"]
        plt.plot(ts, linewidth=2, label=expname)

        # (2) Two model : Evaluate A, tau (without normalization)
        for v in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = run_curve_fit(**common_configs, **expconfig, 
                background_sample_ratio=v, curve_fit_method="two")
            errors["subtract_only_%s"%str(v)] = result["error"]

        # (3) One model : Evaluate tau
        for v in [0.1, 0.3, 0.5, 0.7, 0.9]:#[0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.49]:
            result = run_curve_fit(**common_configs, **expconfig, 
                initial_sample_ratio=0.5*v, background_sample_ratio=0.5*v, 
                curve_fit_method="one")
            errors["normalize_%.2f"%v] = result["error"]
        total_errors[expname] = errors
    
    # save time
    plt.legend()
    plt.savefig("../result/sampled_times.svg", dpi=600)
    plt.savefig("../result/sampled_times.png", dpi=600)
    plt.clf()

    df = pd.DataFrame.from_dict(total_errors)
    min_error = np.min(np.asarray(df))

    df.plot.bar(figsize=(15,6), legend=None)
    plt.axhline(y = min_error, color = 'black', linestyle = '-')
    
    plt.ylim(ymin=0.05, ymax=0.3)
    plt.legend()
    plt.savefig("../result/simulation_full.svg", dpi=600)
    plt.savefig("../result/simulation_full.png", dpi=600)

def run_with_linear(common_configs):
    total_errors = {}
    # tau = common_configs["params"]["tau"]

    expconfig_list = {
        "uniform": {
            "time_sampling_method": "uniform"
        },
        "exponential": {
            "time_sampling_method": "proportional_sample",
            "time_sampling_f": lambda t, tau: np.exp(t / -tau)
        },
        "exponential_derivative": {
            "time_sampling_method": "proportional_sample",
            "time_sampling_f": lambda t, tau: t * np.exp(t / -tau)
        },
        "exponential_derivative_power": {
            "time_sampling_method": "proportional_sample",
            "time_sampling_f": lambda t, tau: np.power(t * np.exp(t / -tau), 10)
        },
        "certain_range_only": {
            "time_sampling_method": "certain_range_only",
            "range_start": 0.99,
            "range_end": 1.01
        }
    }
    
    for expname, expconfig in expconfig_list.items():
        errors = {}
        v = 0.3
        for noise in [0.3, 0.03, 0.003]:
            common_configs_temp = {**common_configs}
            common_configs_temp["noise_stdev"] = noise
            result = run_curve_fit(**common_configs_temp, **expconfig, 
                initial_sample_ratio=v, background_sample_ratio=v, 
                curve_fit_method="one")
            errors["non_linear_%f"%noise] = np.log10(result["error"])

            # (3) One model : Evaluate tau
            result = run_curve_fit(**common_configs_temp, **expconfig, 
                initial_sample_ratio=v, background_sample_ratio=v, 
                curve_fit_method="log_and_linear")
            errors["linear_%f"%noise] = np.log10(result["error"])

        total_errors[expname] = errors

    # plt.show()
    df = pd.DataFrame.from_dict(total_errors)
    patterns = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
    df.plot.bar(figsize=(8,6), legend=None, hatch=['', "//", "", "//", "", "//"])
    plt.axhline(y = 0.0, color = 'black', linestyle = '-')
    plt.ylim(ymin=-3, ymax=0.5)

    plt.savefig("../result/simulation_linear.svg", dpi=600)
    plt.savefig("../result/simulation_linear.png", dpi=600)

def run_with_different_tau(common_configs):
    initial_taus = np.linspace(0.2, 2.0, 100)
    dtau = initial_taus[1] - initial_taus[0]

    # (1) exponential
    errors_exponential = []
    for i in range(len(initial_taus) - 1):
        k = initial_taus[i]
        config = {
            "time_sampling_method": "proportional_sample",
            "time_sampling_f": (lambda t, tau: np.exp(-t / (k * tau)))
        }

        result = run_curve_fit(**common_configs, **config, 
            initial_sample_ratio=0.3, background_sample_ratio=0.3, 
            curve_fit_method="one")
        errors_exponential.append(result["error"])

    # (1) exponential derivative
    errors_derivative = []
    for i in range(len(initial_taus) - 1):
        k = initial_taus[i]
        config = {
            "time_sampling_method": "proportional_sample",
            "time_sampling_f": (lambda t, tau: t * np.exp(-t / (k * tau)))
        }

        result = run_curve_fit(**common_configs, **config, 
            initial_sample_ratio=0.3, background_sample_ratio=0.3, 
            curve_fit_method="one")
        errors_derivative.append(result["error"])

    # (1) exponential derivative power
    errors_derivative_power = []
    for i in range(len(initial_taus) - 1):
        k = initial_taus[i]
        config = {
            "time_sampling_method": "proportional_sample",
            "time_sampling_f": (lambda t, tau: np.power(t * np.exp(-t / (k * tau)), 10))
        }

        result = run_curve_fit(**common_configs, **config, 
            initial_sample_ratio=0.3, background_sample_ratio=0.3, 
            curve_fit_method="one")
        errors_derivative_power.append(result["error"])


    # (2) range only
    errors_range_only = []
    for i in range(len(initial_taus) - 1):
        config = {
            "time_sampling_method": "certain_range_only",
            "range_start": initial_taus[i],
            "range_end": initial_taus[i+1]
        }
        result = run_curve_fit(**common_configs, **config, 
            initial_sample_ratio=0.3, background_sample_ratio=0.3, 
            curve_fit_method="one")
        errors_range_only.append(result["error"])
    
    errors_others = {}
    expconfig_list = {
        "uniform": {
            "time_sampling_method": "uniform"
        }
    }
    for expname, expconfig in expconfig_list.items():
        result = run_curve_fit(**common_configs, **expconfig, 
            initial_sample_ratio=0.3, background_sample_ratio=0.3, 
            curve_fit_method="one")
        errors_others[expname] = result["error"]

    
    plt.axhline(y = errors_others["uniform"], color = 'C0', linestyle = '-')
    plt.plot(initial_taus[0:-1], errors_exponential, color = 'C1', label = "exponential")
    plt.plot(initial_taus[0:-1], errors_derivative, color = 'C2', label = "derivative")
    plt.plot(initial_taus[0:-1], errors_derivative_power, color = 'C3', label = "derivative power")
    plt.plot(initial_taus[0:-1], errors_range_only, color = 'C4', label = "only certain range")
    plt.legend()

    plt.savefig("../result/simulation_changing_initial_tau_guess.svg", dpi=600)
    plt.savefig("../result/simulation_changing_initial_tau_guess.png", dpi=600)

def run_with_hetero_noise(common_configs):
    initial_taus = np.linspace(0.2, 3.0, 100)
    dtau = initial_taus[1] - initial_taus[0]
    
    # homo
    errors_range_only = []
    for i in range(len(initial_taus) - 1):
        config = {
            "time_sampling_method": "certain_range_only",
            "range_start": initial_taus[i],
            "range_end": initial_taus[i+1],
            "force_noiseless": True
        }
        result = run_curve_fit(**common_configs, **config, 
            initial_sample_ratio=0.3, background_sample_ratio=0.3, 
            curve_fit_method="one")
        errors_range_only.append(result["error"])
    plt.plot(initial_taus[0:-1], errors_range_only, color = 'C0', label = "homo")

    # hetero
    errors_range_only_hetero = []
    for i in range(len(initial_taus) - 1):
        config = {
            "time_sampling_method": "certain_range_only",
            "range_start": initial_taus[i],
            "range_end": initial_taus[i+1],
            'hetero': True,
            'hetero_f': lambda x:np.sqrt(x),
            "force_noiseless": True
        }
        result = run_curve_fit(**common_configs, **config, 
            initial_sample_ratio=0.3, background_sample_ratio=0.3, 
            curve_fit_method="one")
        errors_range_only_hetero.append(result["error"])
    
    plt.plot(initial_taus[0:-1], errors_range_only_hetero, color = 'C1', label = "hetero1")

    # hetero
    errors_range_only_hetero = []
    for i in range(len(initial_taus) - 1):
        config = {
            "time_sampling_method": "certain_range_only",
            "range_start": initial_taus[i],
            "range_end": initial_taus[i+1],
            'hetero': True,
            'hetero_f': lambda x:x,
            "force_noiseless": True
        }
        result = run_curve_fit(**common_configs, **config, 
            initial_sample_ratio=0.3, background_sample_ratio=0.3, 
            curve_fit_method="one")
        errors_range_only_hetero.append(result["error"])
    
    # plt.plot(initial_taus[0:-1], errors_range_only, color = 'C1', label = "homo")
    plt.plot(initial_taus[0:-1], errors_range_only_hetero, color = 'C2', label = "hetero2")


    # hetero
    errors_range_only_hetero = []
    for i in range(len(initial_taus) - 1):
        config = {
            "time_sampling_method": "certain_range_only",
            "range_start": initial_taus[i],
            "range_end": initial_taus[i+1],
            'hetero': True,
            'hetero_f': lambda x:1.0/x,
            "force_noiseless": True
        }
        result = run_curve_fit(**common_configs, **config, 
            initial_sample_ratio=0.3, background_sample_ratio=0.3, 
            curve_fit_method="one")
        errors_range_only_hetero.append(result["error"])
    
    # plt.plot(initial_taus[0:-1], errors_range_only, color = 'C1', label = "homo")
    plt.plot(initial_taus[0:-1], errors_range_only_hetero, color = 'C3', label = "hetero3")
    
    plt.ylim(ymin=0.0, ymax=0.3)

    plt.savefig("../result/hetero.svg", dpi=600)
    plt.savefig("../result/hetero.png", dpi=600)

if __name__ == "__main__":
    common_configs = {
        "N": 100,
        "M": 1,
        "n_iter": 1000,
        "start_time": 0.0,
        "end_time": 6.0,
        "noise_stdev": 0.165
    }

    expnumber = sys.argv[1]
    if expnumber == '3' or expnumber == "fig3":
        run_comparison_all(common_configs)
    elif expnumber == '4' or expnumber == "fig4":
        run_with_different_tau(common_configs)
    elif expnumber == '5' or expnumber == "fig5":
        run_with_hetero_noise(common_configs)
    elif expnumber == '6' or expnumber == "fig6":
        run_with_linear(common_configs)
    