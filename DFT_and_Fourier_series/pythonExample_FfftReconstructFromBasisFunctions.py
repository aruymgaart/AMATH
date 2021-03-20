import os,sys,cmath,copy,numpy as np,scipy.io as sio,matplotlib.pyplot as plt 
fft, fftshift, ifft = np.fft.fft, np.fft.fftshift, np.fft.ifft
abs, rndm, exp = np.absolute, np.random.normal, np.exp
machineEpsilon = np.finfo(np.float32).eps
np.set_printoptions(precision=3)

L,n,pi = 10,64,np.pi
Nr = n/2
x2 = np.linspace(-L,L,n+1)
x = x2[0:n]
y,z = x,x
fk = np.append(np.arange(0,n/2),np.arange(-n/2,0))
scale = (2*pi/(2*L)) 
k = scale * fk
ks = fftshift(k)

#-- build a signal from elementary basis waves (from the Fourier basis) --
m1,m2,m3 = 4,9,7
a1,a2 = 2.123, 5.918
b1 = 3.44
s1 = a1*np.cos(m1*pi*x/L)
s2 = a2*np.cos(m2*pi*x/L)
s3 = b1*np.sin(m3*pi*x/L)
signal = s1 + s2 + s3

#-- FFT and peak indices 
st = fft(signal)
stS = fftshift(st)
specIndxs = np.where(abs(st) > machineEpsilon)[0] #-- indexes where FFT nonzero
specIndxsS = np.where(abs(stS) > machineEpsilon)[0]

#-- in this problem we have an FT in space x, not time t
velocity = 1
intervalLength = L - (-L)
wavelength1 = intervalLength/m1
wavelength2 = intervalLength/m2
wavelength3 = intervalLength/m3
freq1 = velocity/wavelength1
freq2 = velocity/wavelength2

sines,cosines = {},{}
for j,p in enumerate(specIndxsS):
	wn, c = ks[p]/scale, stS[p]/Nr
	a, b = abs(c.real), abs(c.imag)
	if a < machineEpsilon : a = 0.0
	if b < machineEpsilon : b = 0.0
	if a != 0 : cosines[abs(wn)] = a
	if b != 0 : sines[abs(wn)] = b

# note: the reconstruction replaces the ifft (this is a "manual" ifft, for educational purpose)

#-- output --
print('wavelength 1', wavelength1, 'freq 1', freq1)
print('wavelength 2', wavelength2, 'freq 2', freq2)
print('locations (index) of peaks in FFT', specIndxs, 'shifted', specIndxsS)
print('wavenumbers of peaks', k[specIndxs[0]]/scale, k[specIndxs[1]]/scale)
print('wavenumbers of peaks (using shifted)', ks[specIndxsS[0]]/scale, ks[specIndxsS[1]]/scale)
print('recovered wav index (n in cos,sin arg)', k[specIndxs[0]]/scale, k[specIndxs[1]]/scale)
print('recovered coefficient', abs(st[specIndxs[0]])/Nr, abs(st[specIndxs[1]])/Nr)
print('BASIS WAVE FUNCTIONS')
rec = np.zeros(n)
for wn in cosines: 
	print('\tCOS', wn, cosines[wn])
	rec += cosines[wn]*np.cos(wn*pi*x/L)
for wn in sines: 
	print('\tSIN', wn, sines[wn])
	rec += sines[wn]*np.sin(wn*pi*x/L)

if False:
	plt.plot(x,s1, label='component wave 1')
	plt.plot(x,s2, label='component wave 2')
	plt.plot(x,s3, label='component wave 3')
plt.plot(x,signal, label='sum of components (signal)')
plt.plot(x,rec+0.2, label='reconstructed') #- add small offset to distinct see curve in plot
plt.axvline(x=wavelength1, c='r')
plt.axvline(x=wavelength2, c='r')
plt.axvline(x=wavelength3, c='r')
plt.legend()
plt.show()

plt.plot(ks,fftshift(np.real(st))/Nr, label='real')
plt.plot(ks,fftshift(np.imag(st))/Nr, label='imag')
plt.legend()
plt.show()




