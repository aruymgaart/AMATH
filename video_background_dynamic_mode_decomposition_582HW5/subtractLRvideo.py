# AP Ruymgaart DMD, main script
import numpy as np, time, sys, copy, matplotlib.pyplot as plt
from videoFunctions import *
from tensorFiles import *
from plottingFunctions import *
from dmd import *

#==== input (command line, from run.sh) ====
print('===================================== start DMD =================================\nInput:')
cmds = processCmdArgs(sys.argv)
for c in cmds: print('\t', c, cmds[c])
dt,nv,fname,ftype,images,f0,f1,outname,frPlot,binv = None,None,None,None,None,None,None,None,[],None
try:	
	nv = int(cmds['modes'])
	dt,thresh = float(cmds['dt']), float(cmds['thresh'])
	fname,ftype,outname = cmds['movie'], cmds['type'], cmds['outname']
	f0,f1 = int(cmds['framestart']), int(cmds['framestop'])
	szplotfr = cmds['plotframes'].split(',')
	binv = cmds['inv'].lower() == 'true'
	for h in szplotfr: frPlot.append(int(h))
except: print('** input error **'), exit()	

if ftype == 'npz': images = tnsrFile2numpy(fname) 
else: images = video2numpy(fname)
print('Movie-shape=', images.shape, 'dt=',dt, 'Nr modes=', nv, 'file-type=', ftype, 'frame', f0, 'to', f1)
print('Plot frames', frPlot, 'output file name', outname)

#==== DMD & dmdDynamics ====
X, szX, szY = flattenVideo(images, f0, f1)
[Phi,A,S,L,X1,X2,b] = DMD(X, szY, szX, nv)
Xdmd, T, omega = dmdDynamics(X1,L,b,Phi,nv,dt=dt)

#==== foreground/background ====
BG = abs(copy.copy(Xdmd.T))
FG = X[0:len(X)-1] - BG + 0.3 #- subtract low rank BG and add a grey background

print(np.min(FG), np.max(FG))
if False:
	R = copy.copy(FG)
	R[R > 0] = 0.0
	FG = FG - R
	BG = BG + R

for n in range(len(FG)): FG[n] = FG[n]/np.max(FG[n])
FG[FG < thresh] = 0.0 # thresholding (see paper)

if False: #- alternative attempt to select modes, not used now
	omegaCut = 0.0
	Xlr,Xhr = np.zeros(Xdmd.shape), np.zeros(Xdmd.shape)
	for k in range(T.shape[1]):
		LRt, HRt = dmdDynamicsLrVec(X1, L, b, Phi, k, dt, omegaCut=omegaCut)	
		Xlr[:,k] = LRt 
		Xhr[:,k] = HRt 
		
	L2 = np.abs(Xlr.T)
	H2 = np.abs(Xhr.T)
	lrMv = reshape2video(L2/np.max(L2), szY, szX)
	np2movieFile(lrMv, outname+'_xlr', invert=binv)
	hrMv = reshape2video(H2/np.max(H2), szY, szX)
	np2movieFile(hrMv, outname+'_xhr', invert=binv)

#==== output ====
plotSV(np.log(np.abs(S) + 1), fname=outname+'_logSV.png')
plotSV(np.abs(omega), fname=outname+'_omega.png')
bgMv = reshape2video(BG, szY, szX)
bgMv = bgMv/np.max(bgMv)
np2movieFile(bgMv, outname+'_LR', invert=binv)
fgMv = reshape2video(FG, szY, szX)
np2movieFile(fgMv, outname+'_diff', invert=binv)
origMv = reshape2video(X[0:len(X)-1], szY, szX)
for fr in frPlot:
	plotFrame(bgMv, fr, outname+'_LR_%d' % (fr))
	plotFrame(fgMv, fr, outname+'_diff_%d' % (fr))
	plotFrame(origMv, fr, outname+'_orig_%d' % (fr))