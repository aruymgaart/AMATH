import numpy as np
from scipy.signal import convolve2d
from skimage.color import rgb2grey
import matplotlib.pyplot as plt
import pickle, copy

def getCanImg(MX,s,fr,x,y,w=10,h=15,offy=1): return MX[s][fr][y+offy-h:y+offy+h,x-w:x+w,:]

def rgbDifference(im1, im2):
	ret = np.ones(im1.shape) * 99999.0
	nI, nJ, mxd = im2.shape[0], im2.shape[1], 0.0
	for i in range(im1.shape[0] - nI):
		for j in range(im1.shape[1] - nJ):
			d = np.sum(np.abs(im1[i:i+nI, j:j+nJ] - im2))
			ret[i,j] = d
			if d > mxd: mxd = d
	ret[ret==99999] = mxd
	return ret/mxd

def imageSetToObjectCoordinates(M, objImg, bConv=False, name='set1', maxD=10.0):
	coords = []
	try: nObj = len(objImg)
	except: objImg = [objImg]
	for k,A in enumerate(M):
		D, pLast, pNew = [], None, None
		for j in range(len(A)):
			dAll,diffs = [],[]
			for no, ob in enumerate(objImg):
				if bConv:
					R = convolve2d(rgb2grey(A[j]), objImg)
					m = np.unravel_index(np.argmax(R), R.shape)
				else:
					R = rgbDifference(A[j], ob)
					m = np.unravel_index(np.argmin(R), R.shape)
				dAll.append( [j, m[1]+ob.shape[1]/2, m[0]+ob.shape[0]/2, no] )
				diffs.append(R[m])			
			if pLast is None:
				pLast = dAll[np.argmin(np.array(diffs))]
				pNew = pLast
			else:	
				pNew = None
				for k, ind in enumerate(np.argsort(np.array(diffs))):
					dist = np.linalg.norm( np.array(dAll[ind])[1:3] - np.array(pLast)[1:3] ) 
					if dist < maxD:	
						pNew = dAll[ind]
						print('\t UPDATING pLast', dAll[ind])
						pLast = copy.copy(dAll[ind])
						break
					else:
						print('skipping index', k, 'dist', dist, 'skipped img', ind)
				if pNew is None: 
					pNew = dAll[np.argmin(np.array(diffs))]	
					dist = np.linalg.norm( np.array(pNew)[1:3] - np.array(pLast)[1:3] )
					pLast = pNew
					print('* no close match, setting', pNew, 'dist', dist )
			D.append(pNew)
		coords.append(np.array(D))
	f = open('%s_coords.dict' % (name), "wb")		
	pickle.dump(coords,f)
	f.close()

def pca(A, nv=3, t=True):
	As = np.zeros(A.shape)
	for k in range(A.shape[0]): As[k,:] = A[k,:] - np.average(A[k,:])
	As /= np.sqrt(float(A.shape[1]-1)) 
	M = As.T if t else As
	[U,S,Vt] = np.linalg.svd(M, full_matrices=False)
	Ured = U[:,0:nv]
	P = np.matmul( np.diag(S), Vt ).T # = np.matmul(M.T, Ured) # same
	return Ured,S,As,P

def plotCanRot(can1,can2,can3,can4,can5,can6,can7,can8,can9,can10,can11,can12,can13,can14,can15,can16,fname=''):
	fig, ax = plt.subplots(4,4)
	for i in range(4):
		for j in range(4): ax[i,j].axis('off')
	ax[0,0].imshow(can1), ax[0,1].imshow(can2), ax[0,2].imshow(can3), ax[0,3].imshow(can4)
	ax[1,0].imshow(can5), ax[1,1].imshow(can6), ax[1,2].imshow(can7), ax[1,3].imshow(can8)
	ax[2,0].imshow(can9), ax[2,1].imshow(can10),ax[2,2].imshow(can11),ax[2,3].imshow(can12)
	ax[3,0].imshow(can13),ax[3,1].imshow(can14),ax[3,2].imshow(can15),ax[3,3].imshow(can16)
	if fname == '': plt.show()
	else: plt.savefig(fname, figsize=(8, 6), dpi=300, bbox_inches='tight')

def plotTrendMatrix(A, names, ttl='', fname=''):
	for j in range(len(A)): plt.plot(A[j], label=names[j])
	if ttl != '': plt.title(ttl)
	plt.xlabel('Frame nr.')
	plt.legend()
	if fname == '': plt.show()
	else: plt.savefig(fname, figsize=(8, 6), dpi=300, bbox_inches='tight')

def plotInvPcProjection(Rs, A, names, nrPC, ttl='Coordinate trajectories recoverd from projection onto first PC'):
	fig, ax = plt.subplots(len(names),1, sharex=True)
	plt.subplots_adjust(wspace=0, hspace=0)
	ax[0].set_title(ttl)
	for k in range(len(A)):
		ax[k].plot(Rs[k], label=names[k]+(' %dPC' % (nrPC))), ax[k].plot(A[k], label=names[k])
		ax[k].set_yticks([0]), ax[k].legend(loc='right')
	ax[len(names)-1].set_xlabel('Frame nr.')
	plt.legend()
	plt.show()

def plotSV(S, fname=''):
	fig, ax = plt.subplots()
	plt.title('Singular values')
	if False: ax.bar(np.arange(len(S)), np.log(S))
	else: ax.bar(np.arange(len(S)), S, color=(0.2, 0.4, 0.6, 0.6), width=0.25)
	plt.savefig('images/exp2_sing_values.png', figsize=(8, 6), dpi=300, bbox_inches='tight')
	xlabs = []
	for j in range(len(S)): xlabs.append('PC%d' % (j+1))
	ax.set_xticks(np.arange(len(S)))
	ax.set_xticklabels(xlabs)
	if fname == '': plt.show()
	else: 
		plt.savefig(fname, figsize=(8, 6), dpi=300, bbox_inches='tight')
		plt.clf()




	









