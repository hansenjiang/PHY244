
# coding: utf-8

# In[86]:


import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from cycler import cycler
from scipy.optimize import curve_fit

#Change default matplotlib rcParams. View available settings via mpl.rcParams.keys()
mpl.rcParams['figure.figsize'] = (10,6)
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['axes.prop_cycle'] = cycler('color', ['#f44242', '#f49841', '#eff248', '#bcf252', '#7cf151', 
                                                   '#5def89', '#5cefc3', '#5ce5ef', '#5fa8e8', '#4658e2',
                                                   '#6a45c6', '#8b44c6', '#b644c6', '#c644a7', '#c64471',
                                                   '#c64450'])
mpl.rcParams['xtick.minor.bottom'] = True
mpl.rcParams['xtick.minor.visible'] = True


# In[132]:


depth = np.loadtxt('data/depth.txt', skiprows=1, delimiter='\t')
depth2 = np.loadtxt('data/depth2.txt', skiprows=1, delimiter='\t')

data = []
for i in [4,7,8]: data.append(np.array([(1/item[0], item[1], 0.01, i) for item in depth if item[2] == i]))
data = np.array(data)

data2 = []
for i in [4,5,6,8,9]: data2.append(np.array([(1/item[0], item[1], 0.01, i) for item in depth2 if item[2] == i]))
data2 = np.array(data2)


# In[84]:


for depth in data:
    plt.errorbar(depth[:,0], depth[:,1], yerr=depth[:,2], elinewidth=1.0, capthick=1.0, capsize=3.0, 
                 fmt='.', label='depth = '+str(depth[0,3])+'mm')
    plt.plot(depth[:,0], depth[:,1], linewidth=0.5)
plt.xlabel('Frequency$^{-1}$ (s)')
plt.ylabel('Average Wavelength (cm)')
plt.title('$\lambda_{av}$ vs $f$')
plt.legend()
plt.savefig('graphs/datar.png')
plt.clf()

for depth in data2:
    plt.errorbar(depth[:,0], depth[:,1], yerr=depth[:,2], elinewidth=1.0, capthick=1.0, capsize=3.0, 
                 fmt='.', label='depth = '+str(depth[0,3])+'mm')
    plt.plot(depth[:,0], depth[:,1], linewidth=0.5)
plt.xlabel('Frequency$^{-1}$ (s)')
plt.ylabel('Average Wavelength (cm)')
plt.title('$\lambda_{av}$ vs $f$')
plt.legend()
plt.savefig('graphs/data2r.png')
plt.clf()


# In[119]:


def linear(x, a, b):
    return a*x + b

def red_chi2(x, y, u, n):
    squares = np.sum(((x-y)/u)**2)
    return squares/(len(x) - n)

opt, opt2 = [], []

for depth in data:
    opt.append((curve_fit(linear, depth[:,0], depth[:,1]/100, sigma=depth[:,2]/100)))
    plt.errorbar(depth[:,0], depth[:,1]/100, yerr=depth[:,2]/100, elinewidth=1.0, capthick=1.0, capsize=3.0, 
                 fmt='.', label='depth = '+str(depth[0,3])+'mm')
    plt.plot(depth[:,0], linear(depth[:,0], opt[-1][0][0], opt[-1][0][1]), linewidth=0.5)
    print(depth[0,3],'mm >','m=',opt[-1][0][0],'+-',np.sqrt(opt[-1][1][0,0]),' c=',opt[-1][0][1],'+-',np.sqrt(opt[-1][1][1,1]))
    print('chi=',red_chi2(depth[:,1]/100, linear(depth[:,0], opt[-1][0][0], opt[-1][0][1]), depth[:,2]/100, 2))
plt.xlabel('Frequency$^{-1}$ (s)')
plt.ylabel('Average Wavelength (m)')
plt.title('$\lambda_{av}$ vs $f$')
plt.legend()
plt.savefig('graphs/datarfit.png')
plt.clf()

print('\n')

for depth in data2:
    opt2.append((curve_fit(linear, depth[:,0], depth[:,1]/100, sigma=depth[:,2]/100)))
    plt.errorbar(depth[:,0], depth[:,1]/100, yerr=depth[:,2]/100, elinewidth=1.0, capthick=1.0, capsize=3.0, 
                 fmt='.', label='depth = '+str(depth[0,3])+'mm')
    plt.plot(depth[:,0], linear(depth[:,0], opt2[-1][0][0], opt2[-1][0][1]), linewidth=0.5)
    print(depth[0,3],'mm >','m=',opt2[-1][0][0],'+-',np.sqrt(opt2[-1][1][0,0]),' c=',opt2[-1][0][1],'+-',np.sqrt(opt2[-1][1][1,1]))
    print('chi=',red_chi2(depth[:,1]/100, linear(depth[:,0], opt2[-1][0][0], opt2[-1][0][1]), depth[:,2]/100, 2))
plt.xlabel('Frequency$^{-1}$ (s)')
plt.ylabel('Average Wavelength (m)')
plt.title('$\lambda_{av}$ vs $f$')
plt.legend()
plt.savefig('graphs/data2rfit.png')
plt.clf()


# In[145]:


v = []
for i in [4,7,8]: 
    v.append(np.array([(item[1]/100, item[1]/100*item[0], 
                        item[1]/100*item[0]*np.sqrt((0.01/item[1]*100)**2+(0.01/item[0])**2), 
                        i) for item in depth if item[2] == i]))
v = np.array(v)

v2 = []
for i in [4,5,6,8,9]: 
    v2.append(np.array([(item[1]/100, item[1]/100*item[0], 
                         item[1]/100*item[0]*np.sqrt((0.01/item[1]*100)**2+(0.01/item[0])**2), 
                         i) for item in depth2 if item[2] == i]))
v2 = np.array(v2)

for level in v:
    plt.errorbar(level[:,0], level[:,1], yerr=level[:,2], elinewidth=1.0, capthick=1.0, capsize=3.0, 
                 fmt='o', label='depth = '+str(level[0,3])+'mm')
plt.xlabel('Wavelength (m)')
plt.ylabel('v (m/s)')
plt.title('$v$ vs $\lambda$')
plt.legend()
plt.savefig('graphs/v.png')
plt.clf()

for level in v2:
    plt.errorbar(level[:,0], level[:,1], yerr=level[:,2], elinewidth=1.0, capthick=1.0, capsize=3.0, 
                 fmt='o', label='depth = '+str(level[0,3])+'mm')
plt.xlabel('Wavelength (m)')
plt.ylabel('v (m/s)')
plt.title('$v$ vs $\lambda$')
plt.legend()
plt.savefig('graphs/v2.png')
plt.clf()


# In[173]:


e4 = np.loadtxt('data/e4.txt', skiprows=1, delimiter='\t')

popt, pcov = curve_fit(linear, 1/e4[:,0], np.sin(e4[:,1]*np.pi/180), sigma=[0.01 for item in e4])
print(popt, np.sqrt(pcov[0,0]), np.sqrt(pcov[1,1]))

plt.errorbar(1/e4[:,0], np.sin(e4[:,1]*np.pi/180), yerr=0.01, elinewidth=1.0, capthick=1.0, capsize=3.0, 
             fmt='.', label='data')
plt.plot(1/e4[:,0], linear(1/e4[:,0], popt[0], popt[1]), linewidth=1.0, label='best fit')
plt.xlabel('a$^{-1}$ (1/mm)')
plt.ylabel('sin($\Theta$)')
plt.title('sin($\Theta$) vs 1/a')
plt.legend()
plt.savefig('graphs/e4.png')


