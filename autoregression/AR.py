# AP Ruymgaart
# https://www.statsmodels.org/stable/examples/notebooks/generated/autoregressions.html
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from tensorFiles import *
import matplotlib.pyplot as plt

data = tnsrFile2numpy('data.npz')
datT = data[:,0:80] # training data (AR fit)

series = 4
ARorder = 8

print('=========== cosine AR(2) ===========')
mod = AutoReg(datT[series], ARorder, old_names=False)
res = mod.fit()
print(dir(res))
print(res.summary())

p = mod.predict(res.params, end=100)
plt.title('AR prediction (right of red line=predicted, left=training)')
plt.plot(p, label='AR(%d) predicted' % (ARorder))
plt.plot(data[series], label='True')
plt.axvline(80,c='r')
plt.legend()
plt.show()
