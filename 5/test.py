from scipy.stats import poisson
from matplotlib import pyplot as plt

n = range(20)
pmf = poisson.pmf(n, 6)

plt.plot(n, pmf)
plt.show()




