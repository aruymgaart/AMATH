# AP Ruymgaart
# https://www.statsmodels.org/stable/examples/notebooks/generated/autoregressions.html
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from tensorFiles import *
import matplotlib.pyplot as plt

series = 4
ARorder = 2
trainEnd = 70

data = tnsrFile2numpy('data.npz')
snames = ['$\\cos(x)$', '$e^{-ax}$', '$e^{ax}$', '$a_1x^5 + a_2x^4 + a_3x^3 + a_4x^2 + a_5 x $', '$\\frac{ 1 - e^{-(p+q)t}  }{  1 + (p/q)e^{-(p+q)t}  }$', '$\\sqrt{x}$']
datT = data[:,0:trainEnd] # training data (AR fit)

print('===================  AR(',ARorder,') ===================')
mod = AutoReg(datT[series], ARorder, old_names=False)
res = mod.fit()
print(res.summary())

p = mod.predict(res.params, end=100)
plt.title('AR prediction (right of red line=predicted, left=training)')
plt.plot(p, label='AR(%d) predicted' % (ARorder))
plt.plot(data[series], label='True (%s)' % (snames[series]))
plt.axvline(trainEnd,c='r')
plt.legend()
plt.savefig('images/AR_%d_%d_%d.png' % (ARorder, series, trainEnd), dpi=200, bbox_inches='tight')
plt.show()
