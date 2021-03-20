import numpy as np
import matplotlib.pyplot as plt
from tensorFiles import *
from pcaF import *
from plottingFunctions import *
from classifier import *

#==== MNIST data ====
M1,Mtest = tnsrFile2numpy('data/mnist/mnist_train_images.npz'), tnsrFile2numpy('data/mnist/mnist_test_images.npz')
M3,L1 = tnsrFile2numpy('data/mnist/mnist_validate_images.npz'), tnsrFile2numpy('data/mnist/mnist_train_labels.npz')
Ltest,L3 = tnsrFile2numpy('data/mnist/mnist_test_labels.npz'),    tnsrFile2numpy('data/mnist/mnist_validate_labels.npz')
Mtrain, Ltrain = np.array(list(M1) + list(M3)), np.array(list(L1)+list(L3)) # 60K training images
ltrain,ltest = [],[]
for vbin in Ltrain: ltrain.append(np.argmax(vbin))
for vbin in Ltest: ltest.append(np.argmax(vbin))
ltest,ltrain = np.array(ltest), np.array(ltrain)
print('Training set', Mtrain.shape, Ltrain.shape, len(ltrain), 'Validation set', Mtest.shape, Ltest.shape, len(ltest))

#==== PCA ====
nEv = 100
useReducedDims = True
Ured,S,P = pca(Mtrain, t=True, nv=nEv)
if useReducedDims:
	Mtrain = np.matmul(Mtrain, Ured)
	Mtest = np.matmul(Mtest, Ured)
	print('TRAIN matrix after PC projection (dim reduce)', Mtrain.shape)

#==== some training and validation sets ====
train_M01, train_l01     = selectDigits([0,1],   Mtrain, ltrain)
train_M12, train_l12     = selectDigits([1,2],   Mtrain, ltrain)
train_M89, train_l89     = selectDigits([8,9],   Mtrain, ltrain)
train_M015,train_l015    = selectDigits([0,1,5], Mtrain, ltrain)
train_M0157,train_l0157  = selectDigits([0,1,5,7], Mtrain, ltrain)
val_M12,    val_l12      = selectDigits([1,2],   Mtest, ltest)
val_M015,   val_l015     = selectDigits([0,1,5], Mtest, ltest)
[train_M12, train_l12]   = randperm([train_M12,  train_l12]) # randomize input order
[train_M89, train_l89]   = randperm([train_M89,  train_l89])
[train_M015,train_l015]  = randperm([train_M015, train_l015])
[train_M0157,train_l0157]= randperm([train_M0157, train_l0157])
[val_M12,   val_l12]     = randperm([val_M12,    val_l12])
[val_M015,  val_l015]    = randperm([val_M015,   val_l015])

#==== LDA ====
LD01 =  lda(train_M01,  train_l01)            # train LDA 2 digits, easy to separate
LD12 =  lda(train_M12,  train_l12)            # train LDA 2 digits
LD89 =  lda(train_M89,  train_l89)            # train LDA 2 digits, hard to separate
LD015 = lda(train_M015, train_l015)           # train LDA 3 digits
LD0157 = lda(train_M0157, train_l0157)        # train LDA 4 digits
LDall = lda(Mtrain, ltrain)                   # train all 10 digits
pred12lda  = LD12.predict(val_M12)            # PREDICT (classify, 2)
pred1015da = LD015.predict(val_M015)          # PREDICT (classify, 3)
predAllLda = LDall.predict(Mtest)             # PREDICT (classify, all 10)
F3l, fLab3l = confusion(pred1015da, val_l015) # Analyze error/misclassification
F10l, fLab10l = confusion(predAllLda, ltest)  # Analyze error/misclassification

#==== SVM ====
SVM12 =  svm(train_M12,  train_l12)           # train SVM 2 digits
SVM015 = svm(train_M015, train_l015)          # train SVM 3 digits
SVM = svm(Mtrain, ltrain)                     # train all 10 digits
predl12svm = SVM12.predict(val_M12)
predl015svm = SVM015.predict(val_M015)
predAllSvm = SVM.predict(Mtest) 
F10s, fLab10s = confusion(predAllSvm, ltest)  # Analyze error/misclassification

#==== DTREE ====
DT12 = dtree(train_M12,  train_l12)
pred12dtree = DT12.predict(val_M12)

#==== output ====
plotSV(S)
plotImageMatrix([Ured[:,0],Ured[:,1],Ured[:,2],Ured[:,3],Ured[:,4],Ured[:,5]],2,3) # first 6 eigendigits
plot3(P[:,0][0:1000], P[:,1][0:1000], P[:,2][0:1000], c=ltrain[0:1000], ttl='All Digits projected on PC 1,2,3', cbar=True, fname='')
plot3(P[:,1][0:1000], P[:,2][0:1000], P[:,3][0:1000], c=ltrain[0:1000], ttl='All Digits projected on PC 2,3,4', cbar=True, fname='')
Rs = np.matmul(train_M12[0:12], Ured.T) #- from PCA compressed space back to original dimensions
plotImageMatrix(Rs,3,4)
Rs = np.matmul(train_M89[0:12], Ured.T) #- from PCA compressed space back to original dimensions
plotImageMatrix(Rs,3,4)
RsAll = np.matmul(P, Ured.T)
plotImageMatrix(RsAll[0:12],3,4)	
plotLDA(LD01.transform(train_M01), train_l01, ttl='LDA of digits 0 & 1')
plotLDA(LD89.transform(train_M89), train_l12, ttl='LDA of digits 8 & 9')
plotLDA(LD015.transform(train_M015), train_l015, ttl='LDA of digits 0,1,5')
plotLDA(LD0157.transform(train_M0157), train_l0157, ttl='LDA of digits 0,1,5,7')	
print('train 1,2', train_M12.shape, 'val 1,2', val_M12.shape)
print('2 digit prediction')
for k in range(10): print('\tpred LDA', pred12lda[k],  'pred SVM', predl12svm[k], 'D Tree', pred12dtree[k], 'true', val_l12[k])
print('3 digit prediction')
for k in range(20): print('\tpred LDA', pred1015da[k], 'pred SVM', predl015svm[k], 'true', val_l015[k])	
print('10 digit prediction')
for k in range(20): print('\tpred LDA', predAllLda[k], 'pred SVM', predAllSvm[k], 'true', ltest[k])	

print('---- 3 digits LDA ----')
print('label indices', fLab3l)
for r in F3l:
	print(r)
correct = np.trace(F3l)
wrong = np.sum(F3l) - correct	
print('Correct', correct, 'wrong', wrong)	

print('---- 10 digits LDA ----')
correct = np.trace(F10l)
wrong = np.sum(F10l) - correct
for r in F10l:
	print(r)
print('Correct', correct, 'wrong', wrong)

print('---- 10 digits SVM ----')
correct = np.trace(F10s)
wrong = np.sum(F10s) - correct
for r in F10s:
	print(r)
print('Correct', correct, 'wrong', wrong)		



