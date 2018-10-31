
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt

#650 nm light


# In[ ]:


#Define graphing constants
page = (19,10)
single_title = 'Light Intensity of Single Slit Diffraction Pattern'
double_title = 'Light Intensity of Double Slit Interference Pattern'


# In[8]:


''' Saves a graph to directory/filename.png with standard options.
'''
def makegraph(x, y, title, filename, color='red', linewidth=0.4, figsize=page, xlabel='Sensor Position (m)',
              ylabel='Light Intensity (V)', directory='graphs/'):
    plt.figure(figsize=figsize)
    plt.plot(x, y, color=color, linewidth=linewidth)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    #plt.savefig(directory+filename+'.png', bbox_inches='tight')
    #plt.clf()

''' Model function for intensity pattern
'''
def intensity(x):
    return 2


# In[9]:


#Load the collected data
single = []
double = []

single.append(np.loadtxt('data/single16a6s10x.txt', skiprows=2))
single.append(np.loadtxt('data/single16a4s1x.txt', skiprows=2))
single.append(np.loadtxt('data/single16a4s100x.txt', skiprows=2))
single.append(np.loadtxt('data/single4a4s100x.txt', skiprows=2))
single.append(np.loadtxt('data/single8a4s100x.txt', skiprows=2))

double.append(np.loadtxt('data/double4a25d4s1x.txt', skiprows=2))
double.append(np.loadtxt('data/double4a25d4s100x.txt', skiprows=2))
double.append(np.loadtxt('data/double4a25d4s100x(2).txt', skiprows=2))
double.append(np.loadtxt('data/double4a50d4s100x.txt', skiprows=2))
double.append(np.loadtxt('data/double8a25d4s100x.txt', skiprows=2))

single = np.array(single)
double = np.array(double)


# In[11]:


#Graph the single slit diffraction patterns
makegraph(single[0][:,0], single[0][:,1], single_title+'a=16mm', 'single16a6s10x')
makegraph(single[1][:,0], single[1][:,1], single_title+'a=16mm', 'single16a4s1x')
makegraph(single[2][:,0], single[2][:,1], single_title+'a=16mm', 'single16a4s100x')
makegraph(single[3][:,0], single[3][:,1], single_title+'a=4mm', 'single4a4s100x')
makegraph(single[4][:,0], single[4][:,1], single_title+'a=8mm', 'single8a4s100x')


# In[12]:


#Graph the double slit interference patterns
makegraph(double[0][:,0], double[0][:,1], double_title+'a=4mm, d=0.25mm', 'double4a25d4s1x')
makegraph(double[1][:,0], double[1][:,1], double_title+'a=4mm, d=0.25mm', 'double4a25d4s100x')
makegraph(double[2][:,0], double[2][:,1], double_title+'a=4mm, d=0.25mm', 'double4a25d4s100x(2)')
makegraph(double[3][:,0], double[3][:,1], double_title+'a=4mm, d=0.50mm', 'double4a50d4s100x')
makegraph(double[4][:,0], double[4][:,1], double_title+'a=8mm, d=0.25mm', 'double8a25d4s100x')

