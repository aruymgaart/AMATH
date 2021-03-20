### import libraries ###
import numpy as np
from mpmath import *
import matplotlib.pyplot as plt
### make MATLAB-like commands for use in Python ###
fft, fftshift, ifft = np.fft.fft, np.fft.fftshift, np.fft.ifft
abs, rndm, exp = np.abs, np.random.normal, np.exp
def sech(x) :  return 1/np.cosh(x)
np.set_printoptions(precision=3)

L,n,pi = 10,64,np.pi
t2 = np.linspace(-L,L,n+1)
t = t2[0:n]
k = (2*pi/(2*L)) * np.append(np.arange(0,n/2),np.arange(-n/2,0)) 

u=sech(t)
ut = fft(u)
noise = 0.5 
rn = rndm(0,noise,[n]) + 1j*rndm(0,noise,[n])
utn = ut + rn
un = ifft(utn)
filter = exp(-0.2*k**2)

ks = fftshift(k) 
fs = fftshift(filter)
utns = fftshift(utn)


### PLOTS ###
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(t,u, label='signal, real space')
ax1.plot(t,un, label='noisy signal, real space (transformed back)')
ax1.plot(t,filter)
ax2.plot(ks, utns/np.max(utns))
ax2.plot(ks, abs(utns)/np.max(utns))
ax2.plot(ks, fs)
plt.savefig('fftNoisySignal.png')

### NOTES ###
'''
u = sech(t) : a simple signal (hyperbolic secant)
ut :
ks : shifted frequencies for k-space plots 
noise : magnitude of noise
rn : COMPLEX noise
'''

### QUESTIONS ###
'''
1) when plot in k space, what part of complex plane are we plotting?
'''