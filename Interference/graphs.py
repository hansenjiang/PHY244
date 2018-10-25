import numpy as np
from matplotlib import pyplot as plt

double = np.loadtxt('data/008a25ddoubleslit4gain100.txt', skiprows=2)

plt.scatter(-double[:,0], double[:,1], s=1**2)
plt.show()


