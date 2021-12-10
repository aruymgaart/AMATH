# AP Ruymgaart
# Test HR pressure from simulation result
import copy, sys, numpy as np, matplotlib.pyplot as plt
from tensorFiles import *

'''
speed sound = 343 m/s  or  0.0029155 s/m
box 0.3m
it takes 0.000875 s for wave to hit back of box.
Bass freq 100Hz = 100/s so a new phase comes every 0.01 s so just over 10 cycles

ds= 0.0011111111111111111 m
Resonator = 120 deep (and 120 wide) x 200 high = 
120*0.0011 = 0.132 m (wide and deep)
200*0.0011 = 0.22  m tall
VOL = 0.132*0.132*0.22 = 0.00383 m^3
AREA TUBE
L= 51*0.0011 = 0.056 m

diam = 39*0.0011 = 0.043 m, radius = 0.0215
A = pi r^2 = 0.0014m^2
freq = (343/2pi) sqrt( 0.0014/0.056*0.00383  )
     = 54.59 sqrt( 6.527 ) = 139.47
     
diam = 19*0.0011 = 0.021 m, radius = 0.0105
A = pi r^2 = 0.000346 
freq = (343/2pi) sqrt( 0.000346/0.056*0.00383  ) 
     = 54.59 sqrt( 1.613 ) = 69.33   
'''          


vU = tnsrFile2numpy('sim_helmholtz_1_5001_RECT-60-179-1-199-05_NONE-0-0-0.npz')
print(vU.shape)

mod = 100
t1, t2, y1, y2, y3, y4 = [], [], [], [], [], []
dt = 0.000002

M = np.zeros((1000,1000))
M[50:150,40:58] = 1
nrElms = int(np.sum(M))
print(nrElms)


for k in range(3,len(vU)):
  M = vU[k]
  p1 = np.sum(M[50:150,40:58])/nrElms
  p2 = np.sum(M[50:150,112:130]) 
  
  if k % mod == 0.0:
  	print(k, p1,p2)
  	y1.append(p1)
  	y2.append(p2)
  	t1.append(k*dt)
  	
  	if False:
  	  tImg = vU[k]/np.max(vU[k])
  	  tImg[50:150,40:58] = 0.25
  	  plt.imshow(tImg)
  	  plt.show()
  	  
  	  
vU = tnsrFile2numpy('sim_helmholtz_2_5001_RECT-60-179-1-199-05_NONE-0-0-0.npz')
print(vU.shape)  


for k in range(3,len(vU)):
  M = vU[k]
  p1 = np.sum(M[50:150,40:58])/nrElms
  p2 = np.sum(M[50:150,112:130]) 
  
  if k % mod == 0.0:
  	print(k, p1,p2)
  	y3.append(p1)
  	y4.append(p2)
  	t2.append(k*dt)
	  
  	  
plt.rcParams.update({'font.size': 22})  	  
plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.size': 22})
#plt.title('Average Acoustic Pressure $u$ in Front Of ') 
plt.plot(t1, y1, c='magenta', label='Helmholtz $1$ (narrower)')
#plt.plot(t1, y2)  	
plt.plot(t2, y3, c='blue', label='Helmholtz $2$ (wider)')
#plt.plot(t2, y4) 
plt.axhline(y=0, c='black')
plt.xlabel('Time $s$')
plt.ylabel('Average Acoustic Pressure $\\bar{u}$')
plt.legend()
plt.savefig("IO/pressure.png", dpi=300, bbox_inches='tight')
#plt.show() 
plt.clf()	