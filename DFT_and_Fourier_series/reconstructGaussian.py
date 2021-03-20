# AP Ruymgaart
# Reconstruct a Gaussian signal from its DFFT
import os,sys,copy,numpy as np,scipy.io as sio,matplotlib.pyplot as plt 
fft, fftshift, ifft = np.fft.fft, np.fft.fftshift, np.fft.ifft
abs, rndm, exp = np.absolute, np.random.normal, np.exp
machineEpsilon = np.finfo(np.float32).eps
np.set_printoptions(precision=3)

def fourierBasis(stS, ks, scale):
	specIndxsS = np.where(abs(stS) > machineEpsilon)[0]
	sines,cosines = {},{}
	Nr = len(stS)
	for j,p in enumerate(specIndxsS):
		n, c = ks[p]/scale, stS[p]/Nr
		a, b = abs(c.real), abs(c.imag)
		if a < machineEpsilon : a = 0.0
		if b < machineEpsilon : b = 0.0
		if a != 0 : cosines[n] = a
		if b != 0 : 
			if n < 0 : sines[n] = -b
			else : sines[n] = b
	return sines,cosines

if __name__ == '__main__':

	L,n,pi = 10,32,np.pi
	x2 = np.linspace(-L,L,n+1)
	x = x2[0:n]
	y,z = x,x
	scale = (2*pi)/(2*L)
	k = scale * np.append(np.arange(0,n/2),np.arange(-n/2,0)) 
	ks = fftshift(k)

	x0 = 0.0
	g = exp(-0.2*(x-x0)**2)
	gt = fft(g)
	gts = fftshift(gt)
	sines,cosines = fourierBasis(gts, ks, scale)

	print('BASIS WAVE FUNCTIONS')
	rec = np.zeros(n)
	for n in cosines: 
		print('\tCOS', n, cosines[n])
		rec += cosines[n]*np.cos(n*scale*x)
	for n in sines: 
		print('\tSIN', n, sines[n])
		rec += sines[n]*np.sin(n*scale*x)

	plt.plot(g)
	plt.plot(rec+0.01) # add some offset to see plot
	plt.show()