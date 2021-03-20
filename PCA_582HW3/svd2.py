import numpy as np, pickle
import matplotlib.pyplot as plt
from imageFunctions import *

coords1 = pickle.load(open('data/cam1_2_coords.dict','rb'))[0]
coords2 = pickle.load(open('data/cam2_2_coords.dict','rb'))[0]
coords3 = pickle.load(open('data/cam3_2_coords.dict','rb'))[0]
x1, x2, x3 = coords1[:,1], coords2[:,1], coords3[:,1]
y1, y2, y3 = coords1[:,2], coords2[:,2], coords3[:,2]
cNames = ['x1','y1','x2','y2','x3','y3']


print(len(x1),len(y1))
print(len(x2),len(y2))
print(len(x3),len(y3))

#- manual align
x2 = x2[4:len(x1)]
y2 = y2[4:len(y1)]
x3 = x3[4:len(x3)]
y3 = y3[4:len(y3)]

'''
plt.plot(y1, label='y1')
plt.plot(y2, label='y2')
plt.plot(x3, label='x3')
plt.legend()
plt.show()
'''

x1, y1 = x1/np.max(y1), y1/np.max(y1)
x2, y2 = x2/np.max(y2), y2/np.max(y2)
x3, y3 = x3/np.max(x3), y3/np.max(x3)
#x1, y1 = x1*2, y1*2

fig, ax = plt.subplots(3,1, sharex=True)
plt.subplots_adjust(wspace=0, hspace=0)
ax[0].set_title('Exp. $2$ normalized aligned coordinate input')
ax[0].plot(x1, label='x1'), ax[0].plot(y1, label='y1'), ax[0].set_yticks([0.4,0.6,0.8]), ax[0].legend()
ax[1].plot(x2, label='x2'), ax[1].plot(y2, label='y2'), ax[1].set_yticks([0.4,0.6,0.8]), ax[1].legend()
ax[2].plot(x3, label='x3'), ax[2].plot(y3, label='y3'), ax[2].set_yticks([0.4,0.6,0.8]), ax[2].legend()
ax[2].set_xlabel("Frame nr.")
plt.legend()
plt.savefig('images/exp2_NONaligned_coordData.png', figsize=(8, 6), dpi=300, bbox_inches='tight')
plt.show()
plt.clf()


#===== cut to same length ====
end = min(len(x1),len(x2),len(x3))
x1, x2, x3 = x1[0:end], x2[0:end], x3[0:end]
y1, y2, y3 = y1[0:end], y2[0:end], y3[0:end]
plotTrendMatrix(np.array([ x1,y1,x2,y2,x3,y3 ]), cNames, ttl='Exp. $2$ aligned coordinate input')
x1, y1 = x1/np.max(y1), y1/np.max(y1)
x2, y2 = x2/np.max(y2), y2/np.max(y2)
x3, y3 = x3/np.max(x3), y3/np.max(x3)
plotTrendMatrix(np.array([ x1,y1,x2,y2,x3,y3 ]), cNames, ttl='Exp. $2$ aligned normalized coordinate input')
#exit()

#==== SVD, PCA ====
A = np.array([ x1,y1,x2,y2,x3,y3 ])
Ured,S,As,P = pca(A, nv=6, t=True)
#As = As.T
Rs2 = np.matmul(P[:,0:2], Ured[:,0:2].T) #- recovered from 3 PC 
Rs3 = np.matmul(P[:,0:3], Ured[:,0:3].T) #- recovered from 3 PC 
Rs4 = np.matmul(P[:,0:4], Ured[:,0:4].T) #- recovered from 4 PC 
Rs5 = np.matmul(P[:,0:5], Ured[:,0:5].T) #- recovered from 5 PC 

fig, ax = plt.subplots(4,1, sharex=True)
plt.subplots_adjust(wspace=0, hspace=0)
ax[0].set_title("Exp. $2$ Principal components")
ax[0].plot(Ured[:,0], label='1st')
ax[1].plot(Ured[:,1], label='2nd')
ax[2].plot(Ured[:,2], label='3rd')
ax[2].plot(Ured[:,3], label='4th')
ax[3].plot(Ured[:,4], label='5th')
ax[3].plot(Ured[:,5], label='6th')
ax[3].set_xlabel('Frame nr.')
for k in range(4): 
	ax[k].legend()
	ax[k].set_yticks([0])
plt.savefig('images/exp2_principal_components.png', figsize=(8, 6), dpi=300, bbox_inches='tight')
plt.legend()
plt.show()
plt.clf()


plotSV(S)
plotInvPcProjection(Rs3, As, cNames, 3, ttl='Coordinate trajectories recoverd from projection onto first $3$ PC')
