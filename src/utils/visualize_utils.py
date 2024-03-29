import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_per_time_graph(eo_delays, total_data):
    #st = 0.001
    #et = 0.010
    #inc = 0.001
    st = 100
    et = 1000
    inc = 100

    target_t = st
    for i in range(len(eo_delays)):
        t = eo_delays[i]
        if t == target_t and target_t <= et:
            target_t += inc
            plt.plot(total_data[:,i], label=t)
    plt.ylim(ymin=-0.0020,ymax=0.0010)
    # for i in range(sx, ex+1, 1):
    #     #, color='r', alpha=(i-sx)/ex
    #     plt.plot(total_data[:,i], label=eo_delays[i])
    plt.axhline(y = 0.0, color='black', linestyle = '-') 
    plt.legend()
    plt.show()
    return

def plot_2d_graph(eo_delays, total_data):
    plt.imshow(total_data, cmap='rainbow', vmin=-0.002, vmax=0.001)
    plt.colorbar()
    plt.show()
    return

def plot_3d_graph(eo_delays, total_data):
    xx, yy = np.meshgrid(range(total_data.shape[1]), range(total_data.shape[0]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, total_data, cmap=cm.coolwarm, linewidth=0)

    plt.show()
    return
