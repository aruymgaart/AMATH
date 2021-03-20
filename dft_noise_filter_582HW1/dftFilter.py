##===== return value lines (one parallel to each axis) through the maximum of F(x,y,z) =====##
def minMaxInfoScalarFuncOfVec(Mesh, F, verbose=False):
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
	if verbose: print('\t', xIndMx,yIndMx,zIndMx, fx[xIndMx], fy[yIndMx], fz[zIndMx], 'at', gx[xIndMx], gy[yIndMx], gz[zIndMx] )
	return [xIndMx,yIndMx,zIndMx,gx,gy,gz,fx,fy,fz]	 

##===== Plot flattened 3D image (into 2D plane) =====##
def flatImage(Z, thresh=0.2, title=None):
  mx = np.max(np.absolute(Z))
  print("image min=",np.min(np.absolute(Z)), "max=", mx, "avg=", np.average(np.absolute(Z)))
  if not thresh is None: Z[Z < thresh*mx] = 0.0
  Im = np.zeros((512,512))
  for i in range(8):
    for j in range(8): 
      m = i + j*8
      Im[i*64:(i+1)*64,j*64:(j+1)*64] = np.absolute(Z[:,:,m])
      Im[:,j*64] = mx
      Im[i*64,:] = mx
  plt.imshow(Im)
  if not title is None: plt.title(title)
  plt.show()

##===== Plot 3 linearized, shared x  =====##
def plotSeriesLinearized(im1,im2,im3): 
  fig, axs = plt.subplots(3, 1, sharex=True)
  fig.subplots_adjust(hspace=0)	
  axs[0].plot(abs(im1.reshape(-1)))
  axs[1].plot(abs(im2.reshape(-1)))
  axs[2].plot(abs(im3.reshape(-1)))
  fig.text(0.85, 0.855, 'n=1', ha='center', va='center')
  fig.text(0.85, 0.600, 'n=24', ha='center', va='center')
  fig.text(0.85, 0.340, 'n=49', ha='center', va='center')
  axs[1].set(ylabel='$|z|$')
  axs[2].set_xlabel('Index')
  plt.show() 
  plt.clf()

##===== Plot 3D trajectory =====##
def plot3dScatter(X):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  a,b,c,d,e,f=-10,10,-10,10,0,10
  ax.axes.set_xlim3d(left=a, right=b) 
  ax.axes.set_ylim3d(bottom=c, top=d) 
  ax.axes.set_zlim3d(bottom=e, top=f) 
  ax.set_xlabel('$x$')
  ax.set_ylabel('$y$')
  ax.set_zlabel('$z$')
  ax.scatter(X[:,0], X[:,1], X[:,2])
  plt.show()

##===== Setup =====##
import os,sys,copy,numpy as np,scipy.io as sio,matplotlib.pyplot as plt 
fft, fftshift, ifft = np.fft.fft, np.fft.fftshift, np.fft.ifft
abs, rndm, exp = np.absolute, np.random.normal, np.exp
np.set_printoptions(precision=3)
try:
  mat_contents = sio.loadmat('data/subdata.mat')
except:
  print('MISSING OR CORRUPT DATA FILE data/subdata.mat (see data/README.txt)')
  exit()
data = mat_contents['subdata']
L,n,pi = 10,64,np.pi
x2 = np.linspace(-L,L,n+1)
x = x2[0:n]
y,z = x,x
k = (2*pi/(2*L)) * np.append(np.arange(0,n/2),np.arange(-n/2,0)) 
ks = fftshift(k)
[X,Y,Z] = np.meshgrid(x,y,z)
[Kx,Ky,Kz] = np.meshgrid(k,k,k)
[Kxs,Kys,Kzs] = np.meshgrid(ks,ks,ks)
im1,im2 = None,None
ftImSeq = []
ftImAvg = np.zeros((n,n,n)).astype('complex')

##===== (III.1) Avg FT image =====##
for j in range(data.shape[1]):
  Un = data[:,j]
  ftIm = np.fft.fftn(Un.reshape(n,n,n)) # FFT 3D
  ftImAvg += ftIm
  ftImSeq.append(ftIm)
  if   j==0 :  im1 = copy.copy(ftImAvg)
  elif j==24 : im2 = copy.copy(ftImAvg)
ftImAvg = ftImAvg/len(ftImSeq)
ftImAvgS = fftshift(ftImAvg)

##===== (III.2) Find the signal (peak) in average FT image =====##
argMxLinS = np.argmax(abs(ftImAvgS))
argMxS = np.unravel_index(argMxLinS, ftImAvg.shape)
pkx,pky,pkz = ks[argMxS[1]], ks[argMxS[0]], ks[argMxS[2]]

##===== (III.3) Gaussian filter in k-space =====##
r = 0.5
filt = exp(-r*(Kx-pkx)**2 - r*(Ky-pky)**2 - r*(Kz-pkz)**2  )

##===== (III.4) Trajectory and LaTex table output =====##
mins, hours, trajectory = 0,0,[]
for j in range(len(ftImSeq)):
  Utn = ftImSeq[j]                # SELECT FT SIGNAL AT t=j
  UnFilt = np.multiply(Utn, filt) # APPLY FILTER
  U = np.fft.ifftn(UnFilt)        # INVERSE FFT 3D
  argMxLin = np.argmax(abs(U))    # PEAK = CENTER OF SUB
  argMx = np.unravel_index(argMxLin, U.shape)
  trajectory.append([ x[argMx[1]], y[argMx[0]], z[argMx[2]] ])
  szTexOut = "$%02d$ (%02d:%02d) & $%3.2f$ & $%3.2f$ & $%3.2f$ \\\\" % \
      (j+1, hours,mins, x[argMx[1]], y[argMx[0]], z[argMx[2]])
  if mins == 0: mins = 30
  else :
    mins = 0
    hours += 1
  print(szTexOut)

##===== plots & other output =====##
print('CENTER WAVENUMBERS', pkx, pky, pkz )
[xIndMxs,yIndMxs,zIndMxs,gxs,gys,gzs,fxs,fys,fzs] = \
  minMaxInfoScalarFuncOfVec([Kxs,Kys,Kzs], abs(ftImAvgS)/np.max(abs(ftImAvgS)))
#flatImage(filt)
#plotSeriesLinearized(im1,im2,ftImAvg)
plot3dScatter(np.array(trajectory))
plt.plot(np.array(trajectory)[:,0])
plt.plot(np.array(trajectory)[:,1])
plt.plot(np.array(trajectory)[:,2])
plt.show()