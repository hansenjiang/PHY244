
# coding: utf-8

# In[36]:


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit
from decimal import Decimal


# In[37]:


#Define graphing constants
page = (19,10)
single_title = 'Light Intensity of Single Slit Diffraction Pattern'
double_title = 'Light Intensity of Double Slit Interference Pattern'

#Global variables
I, offset, length = 1.0, 0.07, 1.0


# In[38]:


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


# In[58]:


''' Model function for single slit intensity pattern
'''
def intensity_s(x, a):
    phi = (np.pi * a)/wavelength * ((x + offset)/np.sqrt((x + offset)**2 + length**2))
    return I * (np.sin(phi)/phi)**2

''' Model function for double slit diffraction pattern
'''
def intensity_d(x, a, d):
    phi_s = (np.pi * a)/wavelength * ((x + offset)/np.sqrt((x + offset)**2 + length**2))
    phi_d = (np.pi * d)/wavelength * ((x + offset)/np.sqrt((x + offset)**2 + length**2))
    return I * (np.cos(phi_d))**2 * (np.sin(phi_s)/phi_s)**2

''' Saves a graph to directory/filename.png with standard options.
'''
def makegraph(x, y, title, filename, color='red', linewidth=0.4, figsize=page, xlabel='Sensor Position (m)',
              ylabel='Light Intensity (V)', directory='graphs/', show_legend=False, label=None,
              plotsecond=False, secondset=(None, None), secondlabel=None):
    plt.figure(figsize=figsize)
    plt.plot(x, y, color=color, linewidth=linewidth, label=label)
    if plotsecond: plt.plot(secondset[0], secondset[1], color='m', linewidth=linewidth+0.6, label=secondlabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if show_legend: plt.legend()
    plt.show()
    #plt.savefig(directory+filename+'.png', bbox_inches='tight')
    plt.clf()

''' Calls makegraph with optimum values and best fit data
'''
def makegraphfit(i, title, filename, which, popt):
    space = np.linspace(-0.13, 0.0, 1000)
    global I
    global offset
    global length
    if (which == 'single'):
        I, offset, length = I_s[i], offset_s[i], length_s
        makegraph(single[i][:,0], single[i][:,1], title, filename, 
                  plotsecond=True, secondset=(space, intensity_s(space, popt)), 
                  show_legend=True, label='data', secondlabel='best fit')
    else:
        I, offset, length = I_d[i], offset_d[i], length_d
        makegraph(double[i][:,0], double[i][:,1], title, filename,
                  plotsecond=True, secondset=(space, intensity_d(space, popt[0], popt[1])), 
                  show_legend=True, label='data', secondlabel='best fit')


# In[90]:


#Load the collected data
single, double, u_s, u_d = [], [], [], []

single.append(np.loadtxt('data/single4a4s100x.txt', skiprows=2))
single.append(np.loadtxt('data/single8a4s100x.txt', skiprows=2))
single.append(np.loadtxt('data/single16a6s10x.txt', skiprows=2))
single.append(np.loadtxt('data/single16a4s1x.txt', skiprows=2))
single.append(np.loadtxt('data/single16a4s100x.txt', skiprows=2))

double.append(np.loadtxt('data/double4a25d4s1x.txt', skiprows=2))
double.append(np.loadtxt('data/double4a25d4s100x.txt', skiprows=2))
double.append(np.loadtxt('data/double4a25d4s100x(2).txt', skiprows=2))
double.append(np.loadtxt('data/double4a50d4s100x.txt', skiprows=2))
double.append(np.loadtxt('data/double8a25d4s100x.txt', skiprows=2))

for i in range(5):
    while (Decimal((float)(single[i][0][0])) == Decimal('0.0')): single[i] = np.delete(single[i], (0), axis=0)
    while (Decimal((float)(double[i][0][0])) == Decimal('0.0')): double[i] = np.delete(double[i], (0), axis=0)
    single, double = np.array(single), np.array(double)
    
for i in range(5): 
    u_s.append([finderror(single[i][j,1], a_slow) for j in range(len(single[i]))])
    u_d.append([finderror(double[i][j,1], a_slow) for j in range(len(double[i]))])
    #for j in range(len(double[i])): double[i][j,0] = np.abs(double[i][j,0])
    
#for i in range(5): u_s.append([600 for j in range(len(single[i]))])
#for i in range(5): u_d.append([60 for j in range(len(double[i]))])

single, double, u_s, u_d = np.array(single), np.array(double), np.array(u_s), np.array(u_d)

wavelength = 650 * 10**(-9)     #650nm (red) light
length_s = 0.783                #distance from the sensor aperture to the laser aperture, single slit
length_d = 0.745                #distance from the sensor aperture to the laser aperture, double slit
I_s = np.array([0.27158, 1.14082, 0.07018, 0.02716, 3.95709])       #maximum intensity in corresponding single data
I_d = np.array([0.00766, 0.17700, 0.67458, 0.71563, 3.30389])       #maximum intensity in corresponding double data
#offset_s = np.array([0.06889, 0.06900, 0.06694, 0.06705, 0.07578])  #offset of maximum in corresponding single data
#offset_d = np.array([0.06939, 0.06755, 0.06939, 0.06817, 0.06928])  #offset of maximum in corresponding double data
offset_s = np.array([0.0694213, 0.0694508, 0.0680048, 0.0675186, 0.0757648])
offset_d = np.array([0.0697878, 0.0679757, 0.0693881, 0.0691909, 0.0692345])


# In[80]:

'''
#Graph the single slit diffraction patterns
makegraph(single[0][:,0], single[0][:,1], single_title+', a=0.04mm', 'single4a4s100x')
makegraph(single[1][:,0], single[1][:,1], single_title+', a=0.08mm', 'single8a4s100x')
makegraph(single[2][:,0], single[2][:,1], single_title+', a=0.16mm', 'single16a6s10x')
makegraph(single[3][:,0], single[3][:,1], single_title+', a=0.16mm', 'single16a4s1x')
makegraph(single[4][:,0], single[4][:,1], single_title+', a=0.16mm', 'single16a4s100x')

#Graph the double slit interference patterns
makegraph(double[0][:,0], double[0][:,1], double_title+', a=0.04mm, d=0.25mm', 'double4a25d4s1x')
makegraph(double[1][:,0], double[1][:,1], double_title+', a=0.04mm, d=0.25mm', 'double4a25d4s100x')
makegraph(double[2][:,0], double[2][:,1], double_title+', a=0.04mm, d=0.25mm', 'double4a25d4s100x(2)')
makegraph(double[3][:,0], double[3][:,1], double_title+', a=0.04mm, d=0.50mm', 'double4a50d4s100x')
makegraph(double[4][:,0], double[4][:,1], double_title+', a=0.08mm, d=0.25mm', 'double8a25d4s100x')


# In[73]:


#Setting up constants for curve fit analysis
popt_s, pcov_s = [0] * 5, [0] * 5
popt_d, pcov_d = [0] * 5, [0] * 5

for i in range(5): 
    I, offset, length = I_s[i], offset_s[i], length_s
    popt_s[i], pcov_s[i] = curve_fit(intensity_s, single[i][:,0], single[i][:,1], sigma=u_s[i])
    I, offset, length = I_d[i], offset_d[i], length_d
    popt_d[i], pcov_d[i] = curve_fit(intensity_d, double[i][:,0], double[i][:,1], sigma=u_d[i])
    
popt_s, pcov_s = np.array(popt_s), np.array(pcov_s)
popt_d, pcov_d = np.array(popt_d), np.array(pcov_d)

print(popt_s, '\n', pcov_s)
print(popt_d, '\n', pcov_d)


# In[91]:

'''

makegraphfit(0, single_title+', a=0.04mm', 'single4a4s100xfit', 'single', 0.04*10**(-3))
makegraphfit(1, single_title+', a=0.08mm', 'single8a4s100xfit', 'single', 0.08*10**(-3))
makegraphfit(0, double_title+', a=0.04mm, d=0.25mm', 'double4a25d4s1xfit', 'double', (0.00004, 0.00025))
makegraphfit(1, double_title+', a=0.04mm, d=0.25mm', 'double4a25d4s100xfit', 'double', (0.00004, 0.00025))
makegraphfit(2, double_title+', a=0.04mm, d=0.25mm', 'double4a25d4s100x(2)fit', 'double', (0.00004, 0.00025))
makegraphfit(3, double_title+', a=0.04mm, d=0.50mm', 'double4a50d4s100xfit', 'double', (0.00004, 0.00050))
makegraphfit(4, double_title+', a=0.08mm, d=0.25mm', 'double8a25d4s100xfit', 'double', (0.00008, 0.00025))

