### import libraries ###
'''
NOTES:
The Fourier transform can be obtained from the complex notation Fourier series
So imaginary components correspond to sines and real components correspond to cosines
'''
import numpy as np
from mpmath import *
import matplotlib.pyplot as plt
### make MATLAB-like commands for use in Python ###
fft, fftshift, ifft = np.fft.fft, np.fft.fftshift, np.fft.ifft
abs, rndm, exp = np.abs, np.random.normal, np.exp
def sech(x) :  return 1/np.cosh(x)
np.set_printoptions(precision=3)

n,m,pi = 32,3,np.pi
L = pi
t2 = np.linspace(-L,L,n+1).astype('float')
t = t2[0:n] 
k = (2*pi/(2*L)) * np.append(np.arange(0,n/2),np.arange(-n/2,0))
ks = fftshift(k)
Cos = np.cos(t*m)
Sin = np.sin(t*m)
FtCos = fft(Cos) 
FtSin = fft(Sin) 

''' MATLAB (same as above)
L=pi; 
n=8;
m=3;
t2=linspace(-L,L,n+1);
t=t2(1:n);
k=(2*pi/(2*L))*[0:(n/2-1) -n/2:-1];
ks = fftshift(k);
Cos=cos(m*t);
Sin=sin(m*t);
FtCos = fft(Cos);
FtSin = fft(Sin);
plot(ks,fftshift(FtCos), ks,fftshift(abs(FtCos)),  ks,fftshift(imag(FtCos)) );
plot(ks,fftshift(FtSin), ks,fftshift(abs(FtSin)),  ks,fftshift(imag(FtSin)) );
xlabel("ks")
'''

f, (ax1, ax2) = plt.subplots(2, 1)
plt.title('$FT(\cos(%dt)), t \in [-\pi,\pi]$' % (m))
ax1.plot(t, Cos, label='$\cos(%dt)$' % (m))
ax1.axvline(x=2*L/m)
ax2.plot(fftshift(k), fftshift(abs(FtCos)), label='abs')
ax2.plot(fftshift(k), fftshift(np.real(FtCos)), label='real')
ax2.plot(fftshift(k), fftshift(np.imag(FtCos)), label='imaginary')
plt.legend()
plt.show()

f, (ax1, ax2) = plt.subplots(2, 1)
plt.title('$FT(\sin(%dt)), t \in [-\pi,\pi]$' % (m))
ax1.plot(t, Cos, label='$\sin(%dt)$' % (m))
ax2.plot(fftshift(k), fftshift(abs(FtSin)), label='abs')
ax2.plot(fftshift(k), fftshift(np.real(FtSin)), label='real')
ax2.plot(fftshift(k), fftshift(np.imag(FtSin)), label='imaginary')
plt.legend()
plt.show()

'''
k
ks
Cos
FtCos
'''
print("\nt",t)
print("\nk",k)
print("\nks",ks)
print("\nCos",Cos)
print("\nFtCos",FtCos)