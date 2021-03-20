import numpy as np, pickle
import matplotlib.pyplot as plt
from imageFunctions import *

coords1 = pickle.load(open('data/cam1_4_coords.dict','rb'))[0] 
coords2 = pickle.load(open('data/cam2_4_coords.dict','rb'))[0] 
coords3 = pickle.load(open('data/cam3_4_coords.dict','rb'))[0]

r1,r2,r3 = coords1[:,3], coords2[:,3], coords3[:,3]
r2 = r2 - 2 #- can image nr offset

#===== shift (modular addition, 16) ====
rn1, rn2, rn3 = [], [], []
for v in r1:
	rn1.append(v+16 if v < 0 else v)
for v in r2:
	rn2.append(v+16 if v < 0 else v)
for v in r3:
	rn3.append(v+16 if v < 0 else v)	
r1, r2, r3 = np.array(rn1), np.array(rn2), np.array(rn3)
x1, y1, z1 = coords1[:,1], coords1[:,2], np.cos(r1*(np.pi/8))
x2, y2, z2 = coords2[:,1], coords2[:,2], np.cos(r2*(np.pi/8))
x3, y3, z3 = coords3[:,1], coords3[:,2], np.cos(r3*(np.pi/8))

#===== manual align ====
F2 = 6
r2, z2, x2, y2 = r2[F2:len(r2)], z2[F2:len(z2)], x2[F2:len(r2)], y2[F2:len(r2)]
	
#===== cut to same length ====
end = min(len(x1),len(x2),len(x3))
x1, x2, x3 = x1[0:end], x2[0:end], x3[0:end]
y1, y2, y3 = y1[0:end], y2[0:end], y3[0:end]
z1, z2, z3 = z1[0:end], z2[0:end], z3[0:end]
r1, r2, r3 = r1[0:end], r2[0:end], r3[0:end]

#==== normalize ====
wz=0.25 # this weight allows projection onto 4 rather than 5 or more PC
x1, y1 = x1/np.max(x1), y1/np.max(x1)
x2, y2 = x2/np.max(x2), y2/np.max(x2)
x3, y3 = x3/np.max(y3), y3/np.max(y3)
z1, z2, z3 = z1*wz, z2*wz, z3*wz

#==== SVD, PCA ====
cNames = ['x1','y1','z1','x2','y2','z2','x3','y3','z3']
A = np.array([ x1,y1,z1,x2,y2,z2,x3,y3,z3 ])
Ured,S,As,P = pca(A, nv=9, t=True)
#As = As.T
Rs2 = np.matmul(P[:,0:2], Ured[:,0:2].T) #- recovered from 3 PC 
Rs3 = np.matmul(P[:,0:3], Ured[:,0:3].T) #- recovered from 3 PC 
Rs4 = np.matmul(P[:,0:4], Ured[:,0:4].T) #- recovered from 4 PC 
Rs5 = np.matmul(P[:,0:5], Ured[:,0:5].T) #- recovered from 5 PC 
Rs6 = np.matmul(P[:,0:6], Ured[:,0:6].T)
Rs7 = np.matmul(P[:,0:7], Ured[:,0:7].T)

print(Ured.shape, P.shape)

#===== plots ====
fig, ax = plt.subplots(9,1, sharex=True)
for k in range(9):
	ax[k].plot(As[k])
plt.show()


fig, ax = plt.subplots(6,1, sharex=True)
plt.subplots_adjust(wspace=0, hspace=0)
ax[0].set_title("Exp. $4$ Principal components")
ax[0].plot(Ured[:,0], label='1st')
ax[1].plot(Ured[:,1], label='2nd')
ax[2].plot(Ured[:,2], label='3rd')
ax[3].plot(Ured[:,3], label='4th')
ax[3].plot(Ured[:,4], label='5th')
ax[4].plot(Ured[:,5], label='6th')
ax[4].plot(Ured[:,6], label='7th')
ax[5].plot(Ured[:,7], label='8th')
ax[5].plot(Ured[:,8], label='9th')
for k in range(6): 
	ax[k].legend()
	ax[k].set_yticks([0])
ax[3].set_xlabel('Frame nr.')
plt.savefig('images/exp4_principal_components.png', figsize=(8, 6), dpi=300, bbox_inches='tight')
plt.legend()
plt.show()
plt.clf()

plotSV(S)
plotInvPcProjection(Rs4, As, cNames, 4, ttl='Coordinate trajectories recoverd from projection onto first $4$ PC')
