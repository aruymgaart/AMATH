import numpy as np
from numpy import sin, cos, pi
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def fun(x, y, z): return cos(x) + cos(y) + cos(z)
def gaussian(X,Y,Z,px,py,pz,w=0.1): return np.exp(-w*(X-px)**2 - w*(Y-py)**2 - w*(Z-pz)**2  )


def isosurface(vol, s=0.1, limits=[-10,10,-10,10,-10,10], labels=['x','y','z']):

	[a,b,c,d,e,f] = limits
	verts, faces, _, _ = measure.marching_cubes_lewiner(vol, spacing=(s, s, s))
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	if True:
		ax.axes.set_xlim3d(left=a, right=b) 
		ax.axes.set_ylim3d(bottom=c, top=d) 
		ax.axes.set_zlim3d(bottom=e, top=f) 
		ax.set_xlabel(labels[0])
		ax.set_ylabel(labels[1])
		ax.set_zlabel(labels[2])
	
	ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], cmap='Spectral', lw=1)
	plt.show()
	
	
# NOTE: gives fist min,max (there need not be a single min or max!)
# also if there are 2 or more minima or maxima but one is just slightly larger
# only that one will show  	
def minMaxInfo(M, Mesh=None, F=None, bAbs=False):

	if bAbs: M = np.absolute(M)

	lmn = np.argmin(M)
	arrIndxs = np.where(M == np.min(M))
	nMin = len(arrIndxs[0])
	if nMin > 1: print('\t* NO SINGLE MIN: Number of points equal to minimum', nMin)
	else: print('\tSINGLE minimum')
	argMn = np.unravel_index(lmn, M.shape)
	
	lmx = np.argmax(M)
	arrIndxs = np.where(M == np.max(M))
	nMax = len(arrIndxs[0])
	if nMin > 1: print('\t* NO SINGLE MAX: Number of points equal to maximum', nMax)
	else: print('\tSINGLE maximum')		
	argMx = np.unravel_index(lmx, M.shape)
	
	print('\tShape', M.shape)
	print('\tFirst min. at', lmn, argMn, 'value', M[argMn], 'First max. at', lmx, argMx, 'value', M[argMx])
	
	return 	[argMn, argMx, np.min(M), np.max(M)]
		

# Scalar valued function of vector position F(x,y,z)
# this function only picks up one maximum (in case there are multiple)
# 
def minMaxInfoScalarFuncOfVec(Mesh, F, verbose=False, bPlot=False):

	if verbose: print('minMaxInfoScalarFuncOfVec')
	
	[X,Y,Z] = Mesh
	
	lmx = np.argmax(abs(F))
	argMx = np.unravel_index(lmx, F.shape)
		
		
	fx,fy,fz = F[argMx[0],:,argMx[2]], F[:,argMx[1],argMx[2]], F[argMx[0],argMx[1],:]	
	gx,gy,gz = X[argMx[0],:,argMx[2]], Y[:,argMx[1],argMx[2]], Z[argMx[0],argMx[1],:]
		
	fMxLin = np.argmax(np.absolute(F))
	argMxF = np.unravel_index(fMxLin, F.shape)
	
	xIndMx = np.argmax(fx)
	yIndMx = np.argmax(fy)
	zIndMx = np.argmax(fz)
	if verbose: print('\t', xIndMx,yIndMx,zIndMx, fx[xIndMx], fy[yIndMx], fz[zIndMx] )
	
	if bPlot:
		f, (ax1, ax2, ax3) = plt.subplots(3, 1)
		ax1.plot(gx, fx, label='x')
		ax2.plot(gy, fy, label='y')
		ax3.plot(gz, fz, label='z')	
		ax1.axvline(x=gx[xIndMx], c='r')
		ax2.axvline(x=gy[yIndMx], c='r')
		ax3.axvline(x=gz[zIndMx], c='r')		
		plt.show()					
	
	return [xIndMx,yIndMx,zIndMx,gx,gy,gz,fx,fy,fz]	

	

if __name__ == '__main__':

	Lx,Ly,Lz,n=10,20,30, 64
	#y,z = x,x	
	x = np.linspace(-Lx,Lx,n) #.astype('complex')
	y = np.linspace(-Ly,Ly,n)
	z = np.linspace(-Lz,Lz,n)
	#y,z = x,x
	[X,Y,Z] = np.meshgrid(x,y,z)
	
	print('---- Mesh grid X info: ----')
	minMaxInfo(X)
	print('\n---- Mesh grid Y info: ---- ')
	minMaxInfo(Y)
	print('\n---- Mesh grid Z info: ---- ') 
	minMaxInfo(Z)
	


	#x, y, z = pi*np.mgrid[-1:1:31j, -1:1:31j, -1:1:31j]
	#vol = fun(x, y, z)
	#isosurface(vol)

	print(x[1]-x[0], 20/64, x[0], x[63])
	F = gaussian(X,Y,Z,5,-7,-2)
	print('\n---- Scalar valued function of vector position G(x,y,z) info: ---- ') 
	minMaxInfoScalarFuncOfVec([X,Y,Z], F, verbose=True, bPlot=True)	
	
	
	argMx = np.unravel_index(np.argmax(abs(F)), F.shape)
	print(argMx, x[argMx[1]], y[argMx[0]], z[argMx[2]])

	F += gaussian(X,Y,Z,8,8,8)
	minMaxInfoScalarFuncOfVec([X,Y,Z], F, verbose=True, bPlot=True)
		
	isosurface(F, s=0.15)

