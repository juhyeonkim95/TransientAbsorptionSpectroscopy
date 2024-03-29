import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv("../data/rc_circuit/data.csv")
data =np.asarray(df)
print(data.shape)
xs = data[:,0]
ys = data[:, 1]
#ys = np.cumsum(data[:, 1])
N = 1000
ys = np.convolve(ys, np.ones(N)/N, mode='same')

plt.plot(xs, ys)
# plt.scatter(xs[::10000], ys[::10000])
#a = np.mean(data[:,1].reshape(-1, 1000), axis=1)
#plt.plot(a)

# data = data[::100,:]
# data = data[700000:700000+200]
#print(data.shape)
#plt.scatter(data[0:100000,0], np.cumsum(data[0:100000,1]))

# plt.scatter(data[:,0], data[:,1])
plt.show()