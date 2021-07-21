# AP Ruymgaart
# simulate ARMA process and plot ACF
import random as rnd, numpy as np
abs, rndm, exp = np.abs, np.random.normal, np.exp
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt



n = 100
noise = 0.5 
rn = rndm(0,noise,[n])
#print(rn)


#=== ARMA(0,0) ===
# just noise 

ACF = acf(rn)

plt.bar(np.arange(len(ACF)), ACF)
plt.title('ARMA(0,0)')
plt.ylabel('Autocorrelation')
plt.xlabel('Lag')
plt.show()


#=== ARMA(1,0) ===
# Y_t = noise_t + a_1 Y_{t-1}
'''
Note: AR(1) with a1=1 is a random walk (same as ARIMA(0,1,0))
'''
a1 = 0.8
Y = []
ym1 = 0.0
for i in range(len(rn)):
	noise_t = rn[i]
	y = noise_t + a1*ym1
	Y.append(y)
	ym1 = y 

ACF = acf(Y)

plt.bar(np.arange(len(ACF)), ACF)
plt.title('ARMA(1,0)')
plt.ylabel('Autocorrelation')
plt.xlabel('Lag')
plt.show()


	

