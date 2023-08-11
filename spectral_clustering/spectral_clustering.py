# AP Ruymgaart
# Spectral clustering example
# Connected regions (graph not fully connected) 

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


f = open('aggr_clust_set_2.csv')
lines = f.readlines()
f.close

data = []
for row in lines:
	elms = row.strip().split('\t')
	print(elms)
	try:
		data.append([float(elms[0]), float(elms[1])])
	except:
		print('error', elms)
data = np.array(data)
sze = len(data)
print('len', sze)

if True:
	plt.scatter(data[:,0], data[:,1])
	plt.show()


#- ADJACENCY MATRIX -
A = np.zeros((sze,sze))
rCut = 2.5
for j in range(sze):
	for i in range(sze):
		d = np.linalg.norm(data[j] - data[i])
		if d < rCut and i != j:
			A[j,i] = 1
			A[i,j] = 1
			#print(d)


if True:
	plt.imshow(A)
	plt.show()


#- DEGREE MATRIX -
Dg = np.zeros((sze,sze))
v = [0] * sze
for n in range(sze):
	nnbrs = np.sum(A[n,:]) 
	v[n] = nnbrs
	Dg[n,n] = nnbrs

#- GRAPH LAPLACIAN
L = Dg - A 

if True:
	plt.imshow(L)
	plt.show()



vals, vecs = np.linalg.eig(L)

# sort these based on the eigenvalues
evecs = vecs[:,np.argsort(vals)]
evals = vals[np.argsort(vals)]

print('sorted eigvals', vals)

# Nr clust is number of (near) zero eigenvectors
s = 0
for v in evals:
	if v < 0.001: s += 1 
k=s

print('NR CONNECTED REGIONS', k) 

# kmeans on first three vectors with nonzero eigenvalues
kmeans = KMeans(n_clusters=k)
#startAt = 
kmeans.fit(evecs[:,0:s])
clust = kmeans.labels_
print('clust', clust)

plt.scatter(data[:,0], data[:,1], c=clust)
plt.show()






