import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
import os
import scipy
from matplotlib import cm
from utils.visualize_utils import *

data_path = "../data"

def drop_invalid_data(df):
    df.set_index('EO delay', inplace=True)
    eo_delays = pd.to_numeric(df.columns, errors = 'coerce')
    invalid_time_indices = np.argwhere(np.isnan(eo_delays)).flatten()
    df = df.drop(df.columns[invalid_time_indices], axis=1)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df


def load_short_data(expname, **kwargs):
    file_name = kwargs.get("file_name", "short.csv")
    df = pd.read_csv(os.path.join(data_path, expname, file_name))
    df = drop_invalid_data(df)
    eo_delays = pd.to_numeric(df.columns, errors = 'coerce')
    df.columns = eo_delays
    
    total_data = np.asarray(df)
    return eo_delays, total_data

def load_unprocessed_data(expname, **kwargs):
    file_name = kwargs.get("file_name", "all.csv")
    df = pd.read_csv(os.path.join(data_path, expname, file_name))
    df.set_index('EO delay', inplace=True)
    columns = np.asarray(df.columns)
    data = np.asarray(df)
    # pump_off_index = columns.index("pump off")
    # print(pump_off_index)
    # print(len(columns))

    N = 99
    M = 126

    pump_off_data = data[:,::N].astype(float)
    pump_on_data = data[:,1::N].astype(float)
    pump_off_idx_name = columns[::N]
    pump_on_t_i = columns[1::N].astype(float)

    if kwargs.get("do_sorting", True):
        t_i_sorted_idx = np.argsort(pump_on_t_i)
        pump_off_data = pump_off_data[:,t_i_sorted_idx]
        pump_on_data = pump_on_data[:,t_i_sorted_idx]
        pump_on_t_i = pump_on_t_i[t_i_sorted_idx]
    print(np.any(np.isnan(pump_on_data)))
    print(np.any(np.isnan(pump_off_data)))
    
    return pump_on_t_i, pump_off_data, pump_on_data

    


def load_long_data(expname, **kwargs):
    file_name = kwargs.get("file_name", "all.csv")
    df = pd.read_csv(os.path.join(data_path, expname, file_name))
    # df = df.rename(columns={"pump off": "-100"})
    # print(df.columns)

    # if "pump off" in df.columns:
    #     pump_off = df.loc[:, "pump off"]
    # else:
    #     pump_off = None

    df = drop_invalid_data(df)

    eo_delays = df.columns[0:]
    eo_delays = np.asarray(eo_delays).astype(float)

    eo_delays_next = eo_delays[1:]
    eo_delays_curr = eo_delays[0:-1]

    eo_delays_increasing_points = np.argwhere(eo_delays_curr > eo_delays_next)
    eo_delays_increasing_points = eo_delays_increasing_points.flatten() + 1

    start_idx = 0

    datas = []
    total_data = np.asarray(df)

    for i in range(len(eo_delays_increasing_points)):
        init_value = eo_delays[start_idx]

        end_idx = eo_delays_increasing_points[i]
        data = total_data[:,start_idx : end_idx]
        times = eo_delays[start_idx : end_idx]
        times_sorted = np.sort(times)
        start_idx = end_idx
        # data = data[:,0] - data[:,-1]
        datas.append([init_value, data])

    datas = sorted(datas, key=lambda x:x[0])
    eo_delays_sorted = list(zip(*datas))[0]
    total_data = list(zip(*datas))[1]
    eo_delays_sorted = np.asarray(eo_delays_sorted)
    total_data = np.asarray(total_data)
    
    total_data = np.transpose(total_data, axes=[2, 1, 0])
    total_data = np.nan_to_num(total_data, nan=0.0, posinf=0.0, neginf=0.0)
    return eo_delays_sorted, total_data