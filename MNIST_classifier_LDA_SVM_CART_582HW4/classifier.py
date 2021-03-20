import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.svm import SVC

def selectDigits(digits, M, labels, sz=None):
	R, l = [],[]
	if sz is None: sz = len(M[0].flatten())
	for d in digits:
		ind = np.argwhere(np.array(labels) == d)
		R += (M[ind].reshape(len(ind),sz)).tolist()
		l += [d] * len(ind)	
	return np.array(R), np.array(l)
	
def randperm(ListArr, nr=None):
	ret = []
	for j in range(len(ListArr)): ret.append([])
	indxs = np.random.permutation(len(ListArr[0]))	
	if not nr is None: indxs = indxs[0:nr]
	for n,i in enumerate(indxs):
		for j in range(len(ListArr)):
			ret[j].append( ListArr[j][i] )
	for j in range(len(ListArr)): ret[j] = np.array(ret[j])			
	return ret
	
def lda(X, labels):
	print('LDA: inputs shape', X.shape, labels.shape)
	clf = LinearDiscriminantAnalysis()
	clf.fit(X, labels)
	print('LDA variance ratio', clf.explained_variance_ratio_)
	return clf

def svm(X, labels):
	clf = SVC(kernel='linear') # Linear Kernel, support vector classifier
	clf.fit(X, labels)
	return clf

def dtree(X, labels, mxDepth=None, mxFeat=None):
	clf = DecisionTreeClassifier(max_depth=mxDepth, max_features=mxFeat) # 
	clf.fit(X, labels)
	return clf

def allPairs(n):
	pairs = []
	for i in range(0,n):
		for j in range(i+1,n):
			pairs.append([i,j])
	return pairs
	
def confusion(predict, true):
	u1, u2 = list(np.unique(predict)), list(np.unique(true))
	ulab = np.sort( np.unique( np.array(u1+u2) ) )	
	F = np.zeros((len(ulab),len(ulab)))
	labIndx = {}
	for n, lb in enumerate(ulab): labIndx[lb] = n
	print(labIndx)
	for k in range(len(predict)):
		i,j = labIndx[predict[k]], labIndx[true[k]]
		F[i,j] += 1
	return F, labIndx			

def adjustImage(im, thresh=0.3):
	im = im - np.min(im)
	mx = np.max(im)
	im[im < thresh*mx] = 0
	return im/np.max(im)	

if __name__ == '__main__':
	print( randperm( [[1,2,3,4,5], [4,5,6,7,8]] ) )

	
