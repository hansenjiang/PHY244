import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from decimal import Decimal
from scipy.stats import chisquare

'''	Returns a value based on a linear curve fit model
'''
def model_function(x, a, b):
	return a*x + b

'''	Returns a value based on a zero-intercept linear curve fit model
'''
def model_zero(x, a):
	return a*x

'''	Finds the first significant digit in Decimal object n.
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

'''	Finds the last significant digit in Decimal object n.
'''
def lastdigit(n):
	place = 0
	while (n % Decimal('1.0') == Decimal('0.0')):
		n = Decimal.shift(n, -1)
	while (n % Decimal('1.0') != Decimal('0.0')):
		n = Decimal.shift(n, 1)
		place -= 1
	return place

'''	Calculates the maximum uncertainty by taking the larger between the error of
	accuracy and the error of precision.
	Error of accuracy is rounded to one significant digit.
'''
def finderror(x, a):
	dec_x = Decimal(str(x))
	dec_a = Decimal(str(a))
	err_a = firstdigit(dec_x * dec_a)
	err_p = Decimal('1.0') * Decimal(str(10.0**(lastdigit(dec_x))))	
	return max(err_a, err_p)

#	Main program

data = np.loadtxt('data.txt', delimiter=',', skiprows=1)
data2 = np.loadtxt('data2.txt', delimiter=',', skiprows=1)

#	The accuracy of the multimeter for voltage and current readings for the appropriate
#	ranges, as provided by the specifications.
a_v = 0.0025
a_c = 0.0075

'''	u, u2 = uncertainties, as calculated by taking the larger between the error of 
	accuracy and error of precision. They correspond to data and data2, respectively. 
	u[:,0] = uncertainties of voltage
	u[:,1] = uncertainties of current
'''
u = np.array([(finderror(data[i,0], a_v), 
				finderror(data[i,1], a_c)) for i in range(len(data))])
u2 = np.array([(finderror(data2[i,0], a_v), 
				finderror(data2[i,1], a_c)) for i in range(len(data2))])

u = np.array([((float)(u[i,0]), (float)(u[i,1])) for i in range(len(u))])
u2 = np.array([((float)(u2[i,0]), (float)(u2[i,1])) for i in range(len(u2))])

p_opt, p_cov = curve_fit(model_function, data[:,0], data[:,1], p0=(1, 0), sigma=u[:,1], 
						absolute_sigma=True)
p_opt2, p_cov2 = curve_fit(model_function, data2[:,0], data2[:,1], p0=(1, 0),
							sigma=u2[:,1], absolute_sigma=True)

print(p_opt[0], p_opt[1])
#	10.075039208093166 -0.001942597419829476
print(p_opt2[0], p_opt2[1])
#	28.37194131404073 1.4894063077357296

print((p_opt[0]/1000)**(-1), (p_opt2[0]/1000)**(-1))
#	99.2551968628282 35.24609010470211

perr = np.sqrt(p_cov[0][0])
perr2 = np.sqrt(p_cov2[0][0])
print(perr/1000 * p_opt[0]**(-2), perr2/1000 * p_opt2[0]**(-2))
#	3.83881435801319e-07 1.6401437585541798e-07		

#	Graph for known resistor = 100 Ohms data
sample = np.linspace(data[0,0], data[-1,0], 500)
plt.plot(sample, model_function(sample, p_opt[0], p_opt[1]), linestyle='--',
			linewidth=0.8)
plt.errorbar(data[:,0], data[:,1], yerr=u[:,1], elinewidth=1.0, capsize=1.8,
				capthick=0.5, ecolor='black', fmt='none')
#plt.plot(data[:,0], (data[:,1] - model_function(data[:,0], p_opt[0], p_opt[1])) * 100)
#plt.plot(data[:,0], [0.886 for i in range(len(data))])
plt.xlabel('Voltage (V)')
plt.ylabel('Current (mA)')
plt.title('Current vs Voltage linear fit, known resistor')
plt.savefig('vvcresistor.png', bbox_inches='tight')
plt.clf()

#	Graph for potentiometer data
sample2 = np.linspace(data2[0,0], data2[-1,0], 500)
plt.plot(sample2, model_function(sample2, p_opt2[0], p_opt2[1]), linestyle='--',
			linewidth=0.8)
plt.errorbar(data2[:,0], data2[:,1], yerr=u2[:,1], elinewidth=1.0, 
				capsize=1.8, capthick=0.5, ecolor='black', fmt='none')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (mA)')
plt.title('Current vs Voltage linear fit, potentiometer')
plt.savefig('vvcpotentiometer.png', bbox_inches='tight')
plt.clf()

p_optz, p_covz = curve_fit(model_zero, data[:,0], data[:,1], p0=(1), sigma=u[:,1], 
						absolute_sigma=True)
p_optz2, p_covz2 = curve_fit(model_zero, data2[:,0], data2[:,1], p0=(1),
							sigma=u2[:,1], absolute_sigma=True)

print(p_optz[0])
#	10.07427367194213
print(p_optz2[0])
#	29.06777613176704

print((p_optz[0]/1000)**(-1), (p_optz2[0]/1000)**(-1))
#	99.26273918735214 34.402356598141644

perrz = np.sqrt(p_covz[0][0])
perrz2 = np.sqrt(p_covz2[0][0])
print(perrz/1000 * p_optz[0]**(-2), perrz2/1000 * p_optz2[0]**(-2))
#	1.6694075379597898e-07 5.734128390395161e-08	

#	Graph for known resistor = 100 Ohms data
plt.plot(sample, model_zero(sample, p_optz[0]), linestyle='--',
			linewidth=0.8)
plt.errorbar(data[:,0], data[:,1], yerr=u[:,1], elinewidth=1.0, capsize=1.8,
				capthick=0.5, ecolor='black', fmt='none')
#plt.plot(data[:,0], (data[:,1] - model_zero(data[:,0], p_optz[0]) * 100)
#plt.plot(data[:,0], [0.886 for i in range(len(data))])
plt.xlabel('Voltage (V)')
plt.ylabel('Current (mA)')
plt.title('Current vs Voltage zero-intercept linear fit, known resistor')
plt.savefig('vvcresistorzero.png', bbox_inches='tight')
plt.clf()

#	Graph for potentiometer data
plt.plot(sample2, model_zero(sample2, p_optz2[0]), linestyle='--',
			linewidth=0.8)
plt.errorbar(data2[:,0], data2[:,1], yerr=u2[:,1], elinewidth=1.0, 
				capsize=1.8, capthick=0.5, ecolor='black', fmt='none')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (mA)')
plt.title('Current vs Voltage linear fit, potentiometer')
plt.savefig('vvcpotentiometerzero.png', bbox_inches='tight')
plt.clf()


#	Section 6 - Reduced chi squared

np.set_printoptions(suppress=True)

def chi_red_squared(N, n, data, function, p_opt, u):
	function_list = [function(data[i,0], p_opt[0], p_opt[1]) for i in range(len(data))]
	sum_terms = [((data[i,1] - function_list[i])/u[i])**2 for i in range(len(data))]
	return 1/(N-n) * sum(sum_terms)

def chi_red_squared_zero(N, n, data, function, p_opt, u):
	function_list = [function(data[i,0], p_opt[0]) for i in range(len(data))]
	sum_terms = [((data[i,1] - function_list[i])/u[i])**2 for i in range(len(data))]
	return 1/(N-n) * sum(sum_terms)

N = len(data)
n = 2

chi = chi_red_squared(N, n, data, model_function, p_opt, u[:,1])
chi2 = chi_red_squared(N, n, data2, model_function, p_opt2, u2[:,1])

print(chi, chi2)
#	0.040376084547311175 4.405122863240309

chiz = chi_red_squared_zero(N, n, data, model_zero, p_optz, u[:,1])
chiz2 = chi_red_squared_zero(N, n, data2, model_zero, p_optz2, u2[:,1])

print(chiz, chiz2)
#	0.040402551723115174 6.188473501288377


	
