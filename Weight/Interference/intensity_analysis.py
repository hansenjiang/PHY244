
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from decimal import Decimal


# In[2]:


#Define graphing constants
page = (12,6)
title_s = 'Light Intensity of Single Slit Diffraction Pattern'
title_d = 'Light Intensity of Double Slit Interference Pattern'


# In[3]:


#Calculate accurate floating point uncertainties using the Decimal module

#Uncertainty percentages recorded from 
#https://www.pasco.com/file_downloads/Downloads_Manuals/PASPORT-High-Sensitivity-Light-Sensor-Manual-PS-2176.pdf
a_slow = 0.0001

''' Finds the first significant digit in Decimal object n.
'''
def firstdigit(n):
    abs_n = abs(n)
    place = 0
    if (abs_n >= Decimal('1.0')):
        while (abs_n >= Decimal('10.0')):
            abs_n = Decimal.shift(abs_n, -1)
            place -= 1
    else:
        while (abs_n < Decimal('1.0')):
            abs_n = Decimal.shift(abs_n, 1)
            place += 1
    return round(n, place)

''' Finds the last significant digit in Decimal object n.
'''
def lastdigit(n):
    place = 0
    while (n % Decimal('1.0') == Decimal('0.0')):
        n = Decimal.shift(n, -1)
    while (n % Decimal('1.0') != Decimal('0.0')):
        n = Decimal.shift(n, 1)
        place -= 1
    return place

''' Calculates the maximum uncertainty by taking the larger between the error of
    accuracy and the error of precision.
    Error of accuracy is rounded to one significant digit.
'''
def finderror(x, a):
    dec_x = Decimal(str(np.abs(x)))
    dec_a = Decimal(str(a))
    err_a = firstdigit(dec_x * dec_a)
    err_p = Decimal('1.0') * Decimal(str(10.0**(lastdigit(dec_x))))
    return (float)(max(err_a, err_p))


# In[45]:


''' Model function for single slit intensity pattern
'''
def intensity_s(x, I, offset, a):
    phi = (np.pi * a)/wavelength * ((x + offset)/np.sqrt((x + offset)**2 + length_s**2))
    return I * (np.sin(phi)/phi)**2

''' Model function for double slit diffraction pattern
'''
def intensity_d(x, I, offset, a, d):
    phi_s = (np.pi * a)/wavelength * ((x + offset)/np.sqrt((x + offset)**2 + length_d**2))
    phi_d = (np.pi * d)/wavelength * ((x + offset)/np.sqrt((x + offset)**2 + length_d**2))
    return I * (np.cos(phi_d))**2 * (np.sin(phi_s)/phi_s)**2

''' Saves a graph to directory/filename.png with standard options.
'''
def makegraph(x, y, title, filename, color='red', linewidth=0.4, figsize=page, xlabel='Sensor Position (m)',
              ylabel='Light Intensity (V)', directory='graphs/', show_legend=False, label=None,
              plotsecond=False, secondset=(None, None), secondlabel=None, u=None):
    plt.figure(figsize=figsize)
    if u == None: plt.plot(x, y, color=color, linewidth=linewidth, label=label)
    else: plt.errorbar(x, y, yerr=u, ecolor=color, elinewidth=0.5, capthick=0.5, capsize=2.0, fmt='none', label=label)
    if plotsecond: plt.plot(secondset[0], secondset[1], color='#197cff', linewidth=linewidth+1.0, label=secondlabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if show_legend: plt.legend()
    plt.savefig(directory+filename+'.png', bbox_inches='tight')
    plt.clf()

''' Calls makegraph with optimum values and best fit data
'''
def makegraphfit(i, title, filename, which, popt, figsize=page, u=None):
    space = np.linspace(-0.13, 0.0, 1000)
    if (which == 'single'):
        makegraph(single[i][:,0], single[i][:,1], title, filename, 
                  plotsecond=True, secondset=(space, intensity_s(space, popt[0], popt[1], popt[2])), 
                  show_legend=True, label='data', secondlabel='best fit', figsize=figsize, u=u)
    else:
        makegraph(double[i][:,0], double[i][:,1], title, filename,
                  plotsecond=True, secondset=(space, intensity_d(space, popt[0], popt[1], popt[2], popt[3])), 
                  show_legend=True, label='data', secondlabel='best fit', figsize=figsize, u=u)


# In[47]:


#Load the collected data
single, double, u_s, u_d = [], [], [], []

single.append(np.loadtxt('data/single4a4s100x.txt', skiprows=2))
single.append(np.loadtxt('data/single8a4s100x.txt', skiprows=2))
single.append(np.loadtxt('data/single16a4s1x.txt', skiprows=2))
single.append(np.loadtxt('data/single16a4s100x.txt', skiprows=2))
single.append(np.loadtxt('data/single16a6s10x.txt', skiprows=2))

double.append(np.loadtxt('data/double4a25d4s1x.txt', skiprows=2))
double.append(np.loadtxt('data/double4a25d4s100x.txt', skiprows=2))
double.append(np.loadtxt('data/double4a25d4s100x2.txt', skiprows=2))
double.append(np.loadtxt('data/double4a50d4s100x.txt', skiprows=2))
double.append(np.loadtxt('data/double8a25d4s100x.txt', skiprows=2))

name_s = ['s4a4s100x', 's8a4s100x', 's16a4s1x', 's16a4s100x', 's16a6s10x']
name_d = ['d4a25d4s1x', 'd4a25d4s100x', 'd4a25d4s100x2', 'd4a50d4s100x', 'd8a25d4s100x']

for i in range(5):
    while (Decimal((float)(single[i][0][0])) == Decimal('0.0')): single[i] = np.delete(single[i], (0), axis=0)
    while (Decimal((float)(double[i][0][0])) == Decimal('0.0')): double[i] = np.delete(double[i], (0), axis=0)
    single, double = np.array(single), np.array(double)
    
for i in range(5): 
    u_s.append([finderror(single[i][j,1], a_slow) for j in range(len(single[i]))])
    u_d.append([finderror(double[i][j,1], a_slow) for j in range(len(double[i]))])

single, double, u_s, u_d = np.array(single), np.array(double), np.array(u_s), np.array(u_d)

wavelength = 650 * 10**(-9)     #650nm (red) light
length_s = 0.783                #distance from the sensor aperture to the laser aperture, single slit
length_d = 0.750                #distance from the sensor aperture to the laser aperture, double slit
I_s = np.array([0.27158, 1.14082, 0.02716, 3.95709, 0.07018])       #maximum intensity in corresponding single data
I_d = np.array([0.00766, 0.17700, 0.67458, 0.71563, 3.30389])       #maximum intensity in corresponding double data
offset_s = np.array([0.0694213, 0.0694508, 0.0675186, 0.0757648, 0.0680048]) #initial guess for positional offset
offset_d = np.array([0.0697878, 0.0679757, 0.0693881, 0.0691909, 0.0692345]) #initial guess for positional offset
a_s = np.array([0.04, 0.08, 0.04, 0.16, 0.04])*10**(-3)
a_d = np.array([[0.04, 0.25], [0.04, 0.25], [0.04, 0.25], [0.04, 0.50], [0.08, 0.25]])*10**(-3)

p0_s = np.array([(I_s[i], offset_s[i], a_s[i]) for i in range(5)])
p0_d = np.array([(I_d[i], offset_d[i], a_d[i,0], a_d[i,1]) for i in range(5)])


# In[14]:


#Setting up constants for curve fit analysis
popt_s, pcov_s = [0] * 5, [0] * 5
popt_d, pcov_d = [0] * 5, [0] * 5

for i in range(5):
    popt_s[i], pcov_s[i] = curve_fit(intensity_s, single[i][:,0], single[i][:,1], p0=p0_s[i], sigma=u_s[i])
    popt_d[i], pcov_d[i] = curve_fit(intensity_d, double[i][:,0], double[i][:,1], p0=p0_d[i], sigma=u_d[i])
    
popt_s, pcov_s = np.array(popt_s), np.array(pcov_s)
popt_d, pcov_d = np.array(popt_d), np.array(pcov_d)

np.set_printoptions(suppress=True)
for i in range(5):
    print(popt_s[i,0],'+-',np.sqrt(pcov_s[i][0,0]),'V')
    print(popt_s[i,1],'+-',np.sqrt(pcov_s[i][1,1]),'m')
    print(popt_s[i,2]*10**3,'+-',np.sqrt(pcov_s[i][2,2])*10**3,'mm')
    print('ratio:',popt_s[i,2]/wavelength,'+-',np.sqrt(pcov_s[i][2,2])/wavelength,'\n')
for i in range(5):
    print(popt_d[i,0],'+-',np.sqrt(pcov_d[i][0,0]),'V')
    print(popt_d[i,1],'+-',np.sqrt(pcov_d[i][1,1]),'m')
    print(popt_d[i,2]*10**3,'+-',np.sqrt(pcov_d[i][2,2])*10**3,'mm')
    print(popt_d[i,3]*10**3,'+-',np.sqrt(pcov_d[i][3,3])*10**3,'mm')
    print('ratio:',popt_d[i,2]/wavelength,'+-',np.sqrt(pcov_d[i][2,2])/wavelength,'\n')


# In[48]:


page_small = (8.5,6)
for i in range(5):
    makegraph(single[i][:,0], single[i][:,1], title_s+', a='+str(a_s[i]*10**3)+'mm', name_s[i], figsize=page_small)
    makegraph(single[i][:,0], single[i][:,1], title_s+', a='+str(a_s[i]*10**3)+'mm', name_s[i]+'e', u=u_s[i])
    makegraphfit(i, title_s+', a='+str(a_s[i]*10**3)+'mm', name_s[i]+'f', 'single', popt_s[i], u=u_s[i])
    makegraphfit(i, title_s+', a='+str(a_s[i]*10**3)+'mm', name_s[i]+'f2', 'single', popt_s[i], u=u_s[i], figsize=page_small)
    makegraphfit(i, title_s+', a='+str(a_s[i]*10**3)+'mm', name_s[i]+'f3', 'single', popt_s[i])
    makegraph(double[i][:,0], double[i][:,1], title_d+', a='+str(a_d[i,0]*10**3)+'mm, d='+str(a_d[i,1]*10**3)+'mm', name_d[i], figsize=page_small)
    makegraph(double[i][:,0], double[i][:,1], title_d+', a='+str(a_d[i,0]*10**3)+'mm, d='+str(a_d[i,1]*10**3)+'mm', name_d[i]+'e', u=u_d[i])
    makegraphfit(i, title_d+', a='+str(a_d[i,0]*10**3)+'mm, d='+str(a_d[i,1]*10**3)+'mm', name_d[i]+'f', 'double', popt_d[i], u=u_d[i])
    makegraphfit(i, title_d+', a='+str(a_d[i,0]*10**3)+'mm, d='+str(a_d[i,1]*10**3)+'mm', name_d[i]+'f2', 'double', popt_d[i], u=u_d[i], figsize=page_small)
    makegraphfit(i, title_d+', a='+str(a_d[i,0]*10**3)+'mm, d='+str(a_d[i,1]*10**3)+'mm', name_d[i]+'f3', 'double', popt_d[i])
    


# In[43]:


#Reduced chi squares
for i in range(5): 
    N_s, N_d, n_s, n_d = len(single[i]), len(double[i]), 3, 4
    func_s = intensity_s(single[i][:,0], popt_s[i,0], popt_s[i,1], popt_s[i,2])
    func_d = intensity_d(double[i][:,0], popt_d[i,0], popt_d[i,1], popt_d[i,2], popt_d[i,3])
    chi_s = np.sum([((single[i][:,1] - func_s)/u_s[i])**2])/(N_s - n_s)
    chi_d = np.sum([((double[i][:,1] - func_d)/u_d[i])**2])/(N_d - n_d)
    print(chi_s,'\n',chi_d,'\n')

