import numpy as np
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import scipy
import pandas as pd


def exponential_with_three_params(t, A, tau, B):
    return A * np.exp(-t / tau) + B

def exponential_with_two_params(t, A, tau):
    return A * np.exp(-t / tau)

def exponential_with_one_params(t, tau):
    return np.exp(-t / tau)

def exponential_with_one_params_derivative(t, tau):
    tau_prime = tau * (1.0 + 0.0 * (2.0 * np.random.random() - 1.0))
    return np.exp(-t / tau_prime) * t

# def exponential_with_one_params_derivative2(t, tau):
#     return np.exp(-t / tau) * t

def t_proportional(t):
    return t

def sample(f, t, M):
    # t_w = 1.0
    # a = np.random.poisson(lam=f_gt(t), size=M)
    # a = a.astype(float)
    # v = np.mean(a) / t_w
    # return v
    return f(t) + np.mean(np.random.normal(0, 0.2, size=M))

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
        sample_f = kwargs.get("time_sample_f")
        fs = sample_f(ts)
        fs /= fs.sum()
        fs_cum = np.cumsum(fs)
        ts = lerp(st, et, fs_cum)
    elif time_sampling_method == "proportional_sample":
        fixed_ts = np.linspace(st, et, 2000)
        sample_f = kwargs.get("time_sample_f")
        fs = sample_f(fixed_ts)
        fs /= fs.sum()
        ts_idx = np.random.choice(len(fixed_ts), N, p=fs)
        ts = fixed_ts[ts_idx]
    ts_new = np.zeros(N+2,)
    ts_new[0] = st
    ts_new[1:N+1] = ts
    ts_new[N+1] = et
    ts = ts_new
    
    return ts

def get_sample_numbers(ts, **kwargs):
    sampling_number_method = kwargs.get("sampling_number_method", "proportional")
    M = kwargs.get("M", 2000)
    N = kwargs.get("N", 100)

    M_total = N * M
    
    if sampling_number_method == "uniform":
        p = np.ones(N)
    elif sampling_number_method == "proportional":
        sample_f = kwargs.get("sample_number_f")
        fs = sample_f(ts)
        fs /= fs.sum()
        p = fs
    p = p.astype(float)
    p /= np.sum(p)
    Ms = M_total * p
    # Ms = Ms.astype(int)
    return Ms


def run_curve_fit(**kwargs):
    n_iter = kwargs.get("n_iter", 1000)
    N = kwargs.get("N", 100)
    parametric_model = kwargs.get("parametric_model")
    gt_params = kwargs.get("gt_params")
    gt_params = np.asarray(gt_params)
    f = lambda t: parametric_model(t, *gt_params)

    fitted_params = []
    noise_mode = kwargs.get("noise_mode", "normal")
    curve_fit_method = kwargs.get("curve_fit_method", "non_linear")

    

    # repetition
    all_ts = []
    for i in trange(n_iter):
        ts = get_time_samples(**kwargs)
        
        ts = np.sort(ts)
        all_ts.append(ts)
        Ms = get_sample_numbers(ts, **kwargs)

        M = kwargs.get("M") 
        total_M = M * (1+N)

        initial_sample_ratio = kwargs.get("initial_sample_ratio", 1.0)
        initial_sample = total_M * (initial_sample_ratio / (initial_sample_ratio + N))
        each_samples = (total_M - initial_sample) * Ms / (M * N)

        # each_sample = (total_M - initial_sample) / N
        # print(initial_sample, each_sample)
        if i == 0:
            print(each_samples)
        
        if noise_mode == "normal":
            noise_sigma = kwargs.get("noise_sigma", 0.1)
            
            observations = f(ts) + np.random.normal(0, noise_sigma, size=len(ts)) * 1.0 / np.sqrt(each_samples)

            # k = kwargs.get("initial_sigma", 1.0)
            if kwargs.get("do_normalization", True):
                observations /= (f(0) + np.random.normal(0, noise_sigma / np.sqrt(initial_sample)))
            # observations /= np.max(observations)
            # observations /= observations[0]
        else:
            observations = []
        
            for n in range(N):
                observation = sample(f, ts[n], Ms[n])
                observations.append(observation)
        try:
            if curve_fit_method == "log_and_linear_regression":
                observations = np.log(observations)

                poly_A, poly_B = np.polyfit(ts, observations, 1)
                if len(gt_params) == 2:
                    params = [np.exp(poly_B), - 1.0 / poly_A]
                else:
                    params = [- 1.0 / poly_A]
            else:
                
                params, _ = scipy.optimize.curve_fit(parametric_model, ts, observations, 
                    #p0=gt_params, sigma=noise_sigma, bounds=(gt_params*0.8, gt_params*1.2)
                )
            fitted_params.append(params)

        except:
            continue
        
    fitted_params = np.asarray(fitted_params)
    print(fitted_params.shape)
    error = np.mean((fitted_params - gt_params[None, :]) ** 2, axis=0)

    print(np.asarray(all_ts).shape)
    ts_mean = np.mean(np.asarray(all_ts), axis=0)

    return error, ts_mean



if __name__ == "__main__":
    errors = {}
    tss = {}
    
    gt_tau = 1.4

    gt_params = [gt_tau]
    params_name = ["tau"]
    parametric_model = exponential_with_one_params

    # gt_params = [1.0, gt_tau]
    # params_name = ["A", "tau"]
    # parametric_model = exponential_with_two_params

    # gt_params = [1.0, gt_tau, 0.1]
    # params_name = ["A", "tau", "B"]
    # parametric_model = exponential_with_three_params

    time_sample_function1 = lambda t: exponential_with_one_params_derivative(t, gt_tau)
    time_sample_function2 = lambda t: exponential_with_one_params(t, gt_tau)
    time_sample_function3 = lambda t:t_proportional(t)
    sample_number_function = lambda t: exponential_with_one_params(t, gt_tau)

    common_configs = {
        "N": 50,
        "M": 200,
        "n_iter": 1000,
        "start_time": gt_tau * 0.0,
        "end_time": gt_tau * 4,
        "time_sampling_method": "proportional_sample",
        "sampling_number_method": "proportional",

        "gt_params": gt_params,
        "parametric_model": parametric_model,
        "noise_sigma": 1.0,
        # "curve_fit_method": "log_and_linear_regression",
        "curve_fit_method": "non_linear",
    }

    common_configs["do_normalization"] = True

    # errors["uniform"] = {}
    # for v in [0.0, 0.5, 1.0, -0.5, -1.0]:
    #     time_sample_f = lambda t: np.power(time_sample_function1(t), 0.0)
    #     sample_number_f = lambda t: np.power(sample_number_function(t), v)
    #     error, _ = run_curve_fit(**common_configs, time_sample_f=time_sample_f, 
    #         sample_number_f=sample_number_f)

    #     errors["uniform"][v] = error[0]
    # df = pd.DataFrame.from_dict(errors)
    # df.plot.bar()
    # plt.show()
    # exit(0)

    # Exp 1. small parts
    # K = 10
    # equal_ts_samples = np.linspace(0.02, 4, K)
    # errors = []
    # for i in range(K-1):
    #     common_configs["start_time"] = equal_ts_samples[i] * gt_tau
    #     common_configs["end_time"] = equal_ts_samples[i + 1] * gt_tau
    #     common_configs["time_sampling_method"] = "uniform"
    #     common_configs["sampling_number_method"] = "uniform"
    #     error, ts = run_curve_fit(**common_configs)
    #     errors.append(error)
    # errors = np.asarray(errors)
    # print(errors.shape)
    # xs = equal_ts_samples[1:]
    # plt.plot(xs, np.log(errors), label=params_name)
    # plt.legend()
    # plt.show()
    # exit(0)

    # Exp2. Unknown tau?
    # for time_sample_power in [0.0, 1.0, 10.0]:
    #     time_sample_f = lambda t: np.power(time_sample_function1(t), time_sample_power)
    #     for sample_number_power in [0.0]:#[0.0, -1.0, 1.0]:
    #         sample_number_f = lambda t: np.power(sample_number_function(t), sample_number_power)
    #         error, ts = run_curve_fit(**common_configs, time_sample_f=time_sample_f, sample_number_f=sample_number_f)

    #         errors["%s_%s" % (str(time_sample_power), str(sample_number_power))] = error
    #         tss["%s_%s" % (str(time_sample_power), str(sample_number_power))] = ts



    # for time_sample_power in [0.0]:
    #     time_sample_f = lambda t: np.power(time_sample_function1(t), time_sample_power)
    #     for sample_number_power in [0.0, -1.0, 1.0]:
    #         sample_number_f = lambda t: np.power(sample_number_function(t), sample_number_power)
    #         error, ts = run_curve_fit(**common_configs, time_sample_f=time_sample_f, sample_number_f=sample_number_f)

    #         errors["%s_%s" % (str(time_sample_power), str(sample_number_power))] = error
    #         tss["%s_%s" % (str(time_sample_power), str(sample_number_power))] = ts

    # for time_sample_power in [0.0, 1.0, 10.0]:
    #     time_sample_f = lambda t: np.power(time_sample_function1(t), time_sample_power)
    #     for sample_number_power in [0.0]:#[0.0, -1.0, 1.0]:
    #         sample_number_f = lambda t: np.power(sample_number_function(t), sample_number_power)
    #         error, ts = run_curve_fit(**common_configs, time_sample_f=time_sample_f, sample_number_f=sample_number_f)

    #         errors["%s_%s" % (str(time_sample_power), str(sample_number_power))] = error
    #         tss["%s_%s" % (str(time_sample_power), str(sample_number_power))] = ts
    

    # for time_sample_power in [0.5, 1.0]:
    #     time_sample_f = lambda t: np.power(time_sample_function2(t), time_sample_power)
    #     for sample_number_power in [0.0]:#[0.0, -1.0, 1.0]:
    #         sample_number_f = lambda t: np.power(sample_number_function(t), sample_number_power)
    #         error, ts = run_curve_fit(**common_configs, time_sample_f=time_sample_f, sample_number_f=sample_number_f)
    #         errors["%s_%s_2" % (str(time_sample_power), str(sample_number_power))] = error
    #         tss["%s_%s_2" % (str(time_sample_power), str(sample_number_power))] = ts


    # for time_sample_power in [-1.0, 1.0]:
    #     time_sample_f = lambda t: np.power(time_sample_function3(t), time_sample_power)
    #     for sample_number_power in [0.0]:#[0.0, -1.0, 1.0]:
    #         sample_number_f = lambda t: np.power(sample_number_function(t), sample_number_power)
    #         error, ts = run_curve_fit(**common_configs, time_sample_f=time_sample_f, sample_number_f=sample_number_f)
    #         errors["%s_%s_3" % (str(time_sample_power), str(sample_number_power))] = error
    #         tss["%s_%s_3" % (str(time_sample_power), str(sample_number_power))] = ts

    errors["uniform"] = {}
    for v in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        time_sample_f = lambda t: np.power(time_sample_function1(t), 0.0)
        sample_number_f = lambda t: np.power(sample_number_function(t), 0.0)
        error, _ = run_curve_fit(**common_configs, time_sample_f=time_sample_f, 
            sample_number_f=sample_number_f,initial_sample_ratio=v)
        errors["uniform"][v] = error[0]

    errors["exponential"] = {}
    for v in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        time_sample_f = lambda t: np.power(time_sample_function2(t), 1.0)
        sample_number_f = lambda t: np.power(sample_number_function(t), 0.0)
        error, _ = run_curve_fit(**common_configs, time_sample_f=time_sample_f, 
            sample_number_f=sample_number_f,initial_sample_ratio=v)
        errors["exponential"][v] = error[0]

    errors["exponential_derivative"] = {}
    for v in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        time_sample_f = lambda t: np.power(time_sample_function1(t), 1.0)
        sample_number_f = lambda t: np.power(sample_number_function(t), 0.0)
        error, _ = run_curve_fit(**common_configs, time_sample_f=time_sample_f, 
            sample_number_f=sample_number_f,initial_sample_ratio=v)
        errors["exponential_derivative"][v] = error[0]
    
    errors["exponential_derivative_power"] = {}
    for v in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        time_sample_f = lambda t: np.power(time_sample_function1(t), 10.0)
        sample_number_f = lambda t: np.power(sample_number_function(t), 0.0)
        error, _ = run_curve_fit(**common_configs, time_sample_f=time_sample_f, 
            sample_number_f=sample_number_f,initial_sample_ratio=v)
        errors["exponential_derivative_power"][v] = error[0]

    

    common_configs["gt_params"] = [1.0, gt_tau]
    common_configs["parametric_model"] = exponential_with_two_params
    common_configs["do_normalization"] = False

    time_sample_f = lambda t: np.power(time_sample_function1(t), 0.0)
    sample_number_f = lambda t: np.power(sample_number_function(t), 0.0)
    error, _ = run_curve_fit(**common_configs, time_sample_f=time_sample_f, 
        sample_number_f=sample_number_f)
    errors["uniform"]["two_parameters"] = error[1]

    time_sample_f = lambda t: np.power(time_sample_function2(t), 1.0)
    sample_number_f = lambda t: np.power(sample_number_function(t), 0.0)
    error, _ = run_curve_fit(**common_configs, time_sample_f=time_sample_f, 
        sample_number_f=sample_number_f)
    errors["exponential"]["two_parameters"] = error[1]
    
    time_sample_f = lambda t: np.power(time_sample_function1(t), 1.0)
    sample_number_f = lambda t: np.power(sample_number_function(t), 0.0)
    error, _ = run_curve_fit(**common_configs, time_sample_f=time_sample_f, 
        sample_number_f=sample_number_f)
    errors["exponential_derivative"]["two_parameters"] = error[1]

    time_sample_f = lambda t: np.power(time_sample_function1(t), 10.0)
    sample_number_f = lambda t: np.power(sample_number_function(t), 0.0)
    error, _ = run_curve_fit(**common_configs, time_sample_f=time_sample_f, 
        sample_number_f=sample_number_f)
    errors["exponential_derivative_power"]["two_parameters"] = error[1]

    df = pd.DataFrame.from_dict(errors)
    print(df)

    df.plot.bar()
    plt.show()
    exit(0)


    # df.set_index('Team').plot.bar(rot = 0)

    # time_sample_f = lambda t: np.power(time_sample_function1(t), 0.0)
    # sample_number_f = lambda t: np.power(sample_number_function(t), 0.0)
    # error, ts = run_curve_fit(**common_configs, time_sample_f=time_sample_f, sample_number_f=sample_number_f)
    # errors["uniform"] = error
    # tss["uniform"] = ts

    # for power in [1.0]:
    #     time_sample_f = lambda t: np.power(time_sample_function2(t), power)
    #     sample_number_f = lambda t: np.power(sample_number_function(t), 0.0)
    #     error, ts = run_curve_fit(**common_configs, time_sample_f=time_sample_f, sample_number_f=sample_number_f)
    #     errors["exponential_%.1f" % power] = error
    #     tss["exponential_%.1f" % power] = ts

    # time_sample_f = lambda t: np.power(time_sample_function1(t), 1.0)
    # sample_number_f = lambda t: np.power(sample_number_function(t), 0.0)
    # error, ts = run_curve_fit(**common_configs, time_sample_f=time_sample_f, sample_number_f=sample_number_f)
    # errors["exponential_derivative"] = error
    # tss["exponential_derivative"] = ts

    # time_sample_f = lambda t: np.power(time_sample_function1(t), 10.0)
    # sample_number_f = lambda t: np.power(sample_number_function(t), 0.0)
    # error, ts = run_curve_fit(**common_configs, time_sample_f=time_sample_f, sample_number_f=sample_number_f)
    # errors["exponential_derivative_power"] = error
    # tss["exponential_derivative_power"] = ts

    # only certain range
    # common_configs["start_time"] = gt_tau * 0.99
    # common_configs["end_time"] = gt_tau * 1.01
    # common_configs["time_sampling_method"] = "uniform"
    # common_configs["sampling_number_method"] = "uniform"
    # error, ts = run_curve_fit(**common_configs)
    # errors["exponential_derivative_inf_power"] = error
    # tss["exponential_derivative_inf_power"] = ts

    for key, item in tss.items():
        plt.plot(item, label=key)
    plt.xlabel("index")
    plt.ylabel("time (gt tau = 1.4)")
    plt.legend()
    plt.show()

    df = pd.DataFrame.from_dict(errors)

    df.index = params_name

    print(df)
    df.plot(kind='bar')
    plt.show()
