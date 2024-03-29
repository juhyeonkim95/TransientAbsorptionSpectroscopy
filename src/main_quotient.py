import numpy as np
import matplotlib.pyplot as plt

N = 10000
a = np.random.normal(3, 0.1, size=N)
b = np.random.normal(4, 0.1, size=N)

c = a / b
plt.hist(c, bins=100)
plt.show()
