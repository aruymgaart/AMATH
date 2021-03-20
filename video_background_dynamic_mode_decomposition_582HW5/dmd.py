# AP Ruymgaart
# Python implementation DMD (Dynamic Mode Decomposition) algorithm 
import numpy as np
import sklearn
import sklearn.utils
import sklearn.utils.extmath
mx,rsvd,svd = np.matmul, sklearn.utils.extmath.randomized_svd, np.linalg.svd
def sech(x) :  return 1/np.cosh(x)

#========== For this function, images should be in rows in X =========
# (P1) Paper Dynamic mode decomposition with control ;  Proctor, Brunton & Kutz
# this is algorithm 1 in Compressed Dynamic Mode Decomposition for Real-Time Object Detection
# Dynamics operator : x_{t+1} = Ax_{t}
# remember matrix multiplication is associative but not commutative
def DMD(X, szY, szX, nv=18, verbose=True):
	X1, X2 = X[0:len(X)-1], X[1:len(X)]
	X1, X2 = X1.T, X2.T                            # images x are now in columns

	if False:
		[U,S,Vt] = rsvd(X1, nv)                    # rsvd seems to not work 
	else:
		[U,S,Vt] = svd(X1, full_matrices=False)    # MATLAB 'econ' mode
		V = Vt.conj().T
		Ur, Sr, Vr = U[:,0:nv], S[0:nv], V[:,0:nv] # Low rank approx

	Sinv = np.diag(1/Sr)   
	VSinv = mx(Vr, Sinv)
	UtX2 = mx(Ur.T, X2)
	A = mx(UtX2, VSinv)                  # eqn. 12 P1 Atilde = Ur'*X2*Vr/Sr;
	L, W = np.linalg.eig(A)              # eigenvecs/vals of A
	Phi = mx( mx(X2, VSinv), W)          # eqn 13 in P1
	b = np.linalg.lstsq(Phi, X1[:,0])[0] # X1[:,0] is first image

	if verbose:
		print('DMD\n\t U from svd', U.shape, 'S', S.shape, 'V* from svd', Vt.shape)
		print('\t reduced shapes', Ur.shape, Sr.shape, Vr.shape, 'Nr PC selected', nv)
		print('\t shapes A', A.shape, 'W', W.shape, 'Phi', Phi.shape, 'b', b.shape)

	return [Phi,A,S,L,X1,X2,b]			

# Build X for all time steps t
# Dynamic Mode Decomposition for Real-Time Background/Foreground Separation in Video
# Grosek & Kutz DMD reconstruct (eqn.5 in paper) 
# x(t) = sum_j b_j phi_j e^{w_j t} where phi_j is a DMD component, b_j is the weight of component phi_j
def dmdDynamics(X1, L, b, Phi, nv=18, dt=0.33, verbose=True):
	omega = np.log(L)/dt         
	T = np.zeros((nv, X1.shape[1])).astype(complex) 
	t = np.arange(X1.shape[1]) * dt
	for k in range(len(t)):
		eO = np.exp( omega * t[k] )
		T[:,k] = b * eO       # Hadamard product
	Xdmd = mx(Phi, T)         # outer : (size_im x nv)(nv x time) = (size_im x time)

	if verbose:
		print('dmdDynamics\n\t T size', T.shape, 'nr frames', len(t), len(b))
		print('\t======== omega =========')
		print('\tomega', omega)
		print('\t|omega|', np.abs(omega))

	return Xdmd, T, omega

# build single time x(t) from low rank sum
# does : x(t) = Phi[:,0]*b[0]*exp(om*t) + Phi[:,1]*b[1]*exp(om*t) + ....	
def dmdDynamicsLrVec(X1, L, b, Phi, nt, dt, omegaCut=0.0, verbose=True):
	omega = np.log(L)/dt  
	t = np.arange(X1.shape[1]) * dt       
	LRt = np.zeros((Phi.shape[0])).astype(complex)
	HRt = np.zeros((Phi.shape[0])).astype(complex)
	for k in range(Phi.shape[1]):
		print('\tmode', k, '|omega|', np.abs(omega[k]), b[k])
		R1 = Phi[:,k] * b[k] * np.exp(omega[k]*t[nt])
		if np.abs(omega[k]) > omegaCut:  HRt += R1
		else:                            LRt += R1				
	return LRt, HRt

if __name__ == "__main__": # DMD test examlple, same as MATLAB dmd.m
	xi = np.linspace(-10,10,400)  	
	t = np.linspace(0,4*np.pi,200)
	dt = t[1] - t[0]
	[Xgrid,T] = np.meshgrid(xi,t)
	f1 = sech(Xgrid+3)*(1.0*np.exp(1j*2.3*T))
	f2 = (sech(Xgrid)*np.tanh(Xgrid))*(2*np.exp(1j*2.8*T))
	f = f1 + f2
	r = 2
	szY, szX, nv = 1, 400, r
	[Phi,A,S,D,X1,X2,b] = DMD(f, szY, szX, r)
	Z, T, omega = dmdDynamics(X1, D, b, Phi, r, dt)
	print('A', A)
	print('b', b)
	print(Z[0:3,0:3])