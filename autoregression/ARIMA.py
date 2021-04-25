# AP Ruymgaart
# https://www.statsmodels.org/stable/examples/notebooks/generated/autoregressions.html
import statsmodels as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
from tensorFiles import *
'''
This will also do special cases
    * autoregressive models: AR(p)
    * moving average models: MA(q)
    * mixed autoregressive moving average models: ARMA(p, q)
    * integration models: ARIMA(p, d, q)
    * seasonal models: SARIMA(P, D, Q, s)
    * regression with errors that follow one of the above ARIMA-type models

parameters
	The (p,d,q) order of the model for the autoregressive, differences, and moving average components. 
	d is always an integer, while p and q may either be integers or lists of integers.

	The (P,D,Q,s) order of the seasonal component of the model for the AR parameters, differences, MA parameters, and periodicity. 
	Default is (0, 0, 0, 0). D and s are always integers, while P and Q may either be integers or lists of positive integers
'''

series = 8 # dataN[8] is x^2  so should have I=2 to make stationary
order1,order2,order3 = 0,2,0
trainEnd = 70
plotACFs = False

data = tnsrFile2numpy('dataN.npz')
dataTrends = tnsrFile2numpy('data.npz')
snames = ['$\\cos(x)$', '$e^{-ax}$', '$e^{ax}$', '$a_1x^5 + a_2x^4 + a_3x^3 + a_4x^2 + a_5 x $', '$\\frac{ 1 - e^{-(p+q)t}  }{  1 + (p/q)e^{-(p+q)t}  }$', '$\\sqrt{x}$', '$ax$', '$0$', '$x^2$']
datT = data[:,0:trainEnd]

if plotACFs:
	for s in range(len(data)):
		vacf = acf(datT[s])
		plt.plot(vacf, label=snames[s])
	plt.title("Autocorrelation function ACF")
	plt.legend()
	plt.savefig('ACF.png', dpi=200, bbox_inches='tight')
	exit()

differenced = None
if order2 > 0:
	differenced = diff(data[series], k_diff=2)

mod = ARIMA(datT[series], order=(order1, order2, order3))
res = mod.fit()
print(res.summary())

p = mod.predict(res.params, end=100)
plt.title('ARIMA prediction (right of red line=predicted, left=training)')
plt.plot(p, label='ARIMA, order=(%d,%d,%d) predicted' % (order1, order2, order3))
plt.plot(data[series], label='True (%s)' % (snames[series]))
plt.plot(dataTrends[series], label='True (%s) trend (denoised)' % (snames[series]))
if not differenced is None: 
	plt.plot(differenced, label='Differenced')
	plt.plot(data[series]-dataTrends[series], label='Random component' )
plt.axvline(trainEnd,c='r')
plt.legend()
plt.savefig('images/ARIMA_%d-%d-%d_%d_%d.png' % (order1, order2, order3, series, trainEnd), dpi=200, bbox_inches='tight')
plt.show()