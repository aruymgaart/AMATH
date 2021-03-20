import numpy as np
import sklearn
import sklearn.utils
import sklearn.utils.extmath

def pca(A, t=True, nv=3):
	As = np.zeros(A.shape)
	for k in range(A.shape[0]): As[k,:] = A[k,:] - np.average(A[k,:])
	As /= np.sqrt(float(A.shape[1]-1)) 
	M = As.T if t else As	
	[U,S,Vt] = sklearn.utils.extmath.randomized_svd(M, nv)
	Ured = U[:,0:nv]
	P = np.matmul( np.diag(S), Vt ).T #P = np.matmul(A.T, Ured)
	print('input A shape',M.shape, 'U', U.shape, 'S', S.shape, 'Vt', Vt.shape, 'P', P.shape, 'Ured', Ured.shape)
	return Ured,S,P