import pickle, numpy as np
import matplotlib.pyplot as plt
from tensorFiles import *
from pcaF import *
from plottingFunctions import *
from classifier import *
#=== MNIST data ===
M1,Mtest = tnsrFile2numpy('data/mnist/mnist_train_images.npz'), tnsrFile2numpy('data/mnist/mnist_test_images.npz')
M3,L1 = tnsrFile2numpy('data/mnist/mnist_validate_images.npz'), tnsrFile2numpy('data/mnist/mnist_train_labels.npz')
Ltest,L3 = tnsrFile2numpy('data/mnist/mnist_test_labels.npz'),    tnsrFile2numpy('data/mnist/mnist_validate_labels.npz')
Mtrain, Ltrain = np.array(list(M1) + list(M3)), np.array(list(L1)+list(L3)) # 60K training images
ltrain,ltest = [],[]
for vbin in Ltrain: ltrain.append(np.argmax(vbin))
for vbin in Ltest: ltest.append(np.argmax(vbin))
print(Mtrain.shape, Mtest.shape)

#=== PCA ===
if True:
	nEv = 10
	Ured,S,P = pca(Mtrain, nv=nEv)
	Mtrain = np.matmul(Mtrain, Ured)
	Mtest = np.matmul(Mtest, Ured)
	print('TRAIN matrix after projection ', Mtrain.shape)

#=== pairwise ===
pairs = allPairs(10)
E,Etr = {},{}
for SEL in [1,2,3]:
	try: E[SEL]
	except: E[SEL] = []
	try: Etr[SEL]
	except: Etr[SEL] = []
	for pair in pairs:
		train_M, train_l   = selectDigits(pair, Mtrain, ltrain)#, sz=sz)
		[train_M, train_l] = randperm([train_M, train_l])
		val_M, val_l = selectDigits(pair,   Mtest, ltest)#, sz=sz)
		if   SEL == 1: Model = lda(train_M,  train_l)
		elif SEL == 2: Model = svm(train_M,  train_l)
		elif SEL == 3: Model = dtree(train_M,  train_l, mxDepth=None)

		predict = Model.predict(train_M)
		nrErrTrain = len(np.nonzero(train_l - predict)[0])
		predict = Model.predict(val_M)
		nrErrEval = len(np.nonzero(val_l - predict)[0])

		E[SEL].append(  [pair, nrErrEval, len(val_l), nrErrTrain, len(train_l) ] )
		print(SEL, '** PAIR =', pair, 'EVAL err', nrErrEval,len(val_l), 'Train error', nrErrTrain, len(train_l))

#=== output ===		
for e in E:
	for k in E[e] : print(e, k)
f = open('data/pairwiseError.dict', "wb")		
pickle.dump(E,f)
f.close()
