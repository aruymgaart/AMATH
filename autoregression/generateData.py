# AP Ruymgaart, generate some example functions
import numpy as np
import copy
import matplotlib.pyplot as plt
from tensorFiles import *

X = np.linspace(0,6*np.pi,100)

# deterministic time series
cosx = np.cos(X)
expxm = np.exp(-0.5*X)
expxp = np.exp(0.5*X)
polyx5 = 0.002*X**5 - 0.045*X**4 + 0.2*X**3 + 0.1*X**2 + 0.5*X
bass1 = ( 1 - np.exp(-(0.01+0.5)*X) )/(1 + (0.5*np.exp(-(0.01+0.5)*X)/0.01 ) )
sqrtx = np.sqrt(X)
linx = 0.3*X
statx = 0.0*X
x2 = X**2

data = [cosx/np.max(cosx), expxm/np.max(expxm), expxp/np.max(expxp), polyx5/np.max(polyx5), bass1/np.max(bass1), sqrtx/np.max(sqrtx), linx/np.max(linx), statx, x2/np.max(x2)]
numpy2tnsrFile(np.array(data), 'data.npz')

plt.title('Sample functions')
plt.ylabel('$f(x)$')
plt.xlabel('x')
plt.plot(X, data[0], label='$\\cos(x)$')
plt.plot(X, data[1], label='$e^{-ax}$')
plt.plot(X, data[2], label='$e^{ax}$')
plt.plot(X, data[3], label='$a_1x^5 + a_2x^4 + a_3x^3 + a_4x^2 + a_5 x $')
plt.plot(X, data[4], label='$\\frac{ 1 - e^{-(p+q)t}  }{  1 + (p/q)e^{-(p+q)t}  }$')
plt.plot(X, data[5], label='$\\sqrt{x}$')
plt.plot(X, data[6], label='$ax$')
plt.plot(X, data[7], label='$0$')
plt.plot(X, data[8], label='$x^2$')
plt.legend()
plt.savefig('sample_functions.png', dpi=200, bbox_inches='tight')
plt.show()

# now turn into stochastic processes by adding noise
rndn = np.random.normal
am = [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
dataN = copy.copy(data)
for k in range(len(dataN)): dataN[k] += rndn(0,am[k],[len(X)])
numpy2tnsrFile(np.array(dataN), 'dataN.npz')

plt.title('Sample functions')
plt.ylabel('$f(x)$')
plt.xlabel('x')
plt.plot(X, dataN[0], label='$\\cos(x)$')
plt.plot(X, dataN[1], label='$e^{-ax}$')
plt.plot(X, dataN[2], label='$e^{ax}$')
plt.plot(X, dataN[3], label='$a_1x^5 + a_2x^4 + a_3x^3 + a_4x^2 + a_5 x $')
plt.plot(X, dataN[4], label='$\\frac{ 1 - e^{-(p+q)t}  }{  1 + (p/q)e^{-(p+q)t}  }$')
plt.plot(X, dataN[5], label='$\\sqrt{x}$')
plt.plot(X, dataN[6], label='$ax$')
plt.plot(X, dataN[7], label='$0$')
plt.plot(X, dataN[8], label='$x^2$')
plt.legend()
plt.savefig('sample_functionsN.png', dpi=200, bbox_inches='tight')
plt.show()