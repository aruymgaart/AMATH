# AP Ruymgaart
# https://www.statsmodels.org/stable/examples/notebooks/generated/autoregressions.html
import statsmodels as sm
from statsmodels.tsa.arima_model import ARIMA
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
'''

series = 4
order1,order2,order3 = 1,0,0
trainEnd = 70

data = tnsrFile2numpy('data.npz')
snames = ['$\\cos(x)$', '$e^{-ax}$', '$e^{ax}$', '$a_1x^5 + a_2x^4 + a_3x^3 + a_4x^2 + a_5 x $', '$\\frac{ 1 - e^{-(p+q)t}  }{  1 + (p/q)e^{-(p+q)t}  }$', '$\\sqrt{x}$']
datT = data[:,0:trainEnd]

mod = ARIMA(datT[series], order=(order1, order2, order3))
res = mod.fit()
print(res.summary())

p = mod.predict(res.params, end=100)
plt.title('ARIMA prediction (right of red line=predicted, left=training)')
plt.plot(p, label='ARIMA, order=(%d,%d,%d) predicted' % (order1, order2, order3))
plt.plot(data[series], label='True (%s)' % (snames[series]))
plt.axvline(trainEnd,c='r')
plt.legend()
plt.savefig('images/ARIMA_%d-%d-%d_%d_%d.png' % (order1, order2, order3, series, trainEnd), dpi=200, bbox_inches='tight')
plt.show()