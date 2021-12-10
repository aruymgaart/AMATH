# AP Ruymgaart
# generate boundaries for PDE solver
import copy, sys, numpy as np, matplotlib.pyplot as plt
from pde_findiff_functions import *


def boundary2rgb(B, bInvertBG=True):
  RGB = np.zeros( (B.shape[0], B.shape[1], 3) )
  Rd, Gr, Bl = np.zeros(B.shape), np.zeros(B.shape), np.zeros(B.shape)
  
  Gr[B == REFLECT_TOP]    = 1  
  Rd[B == REFLECT_BOTTOM] = 1
  Bl[B == REFLECT_LEFT]   = 1
  Gr[B == REFLECT_RIGHT], Rd[B == REFLECT_RIGHT] = 1, 1 # yellow

  TR, BR = REFLECT_TOP + REFLECT_RIGHT, REFLECT_BOTTOM + REFLECT_RIGHT
  TL, BL = REFLECT_TOP + REFLECT_LEFT, REFLECT_BOTTOM + REFLECT_LEFT
  
  Rd[B == TR], Gr[B == TR], Bl[B == TR] = 0.8, 0.5, 0.3
  Rd[B == BR], Gr[B == BR], Bl[B == BR] = 0.2, 0.7, 0.9
  
  Rd[B == TL], Gr[B == TL], Bl[B == TL] = 1.0, 0.0, 1.0 # magenta
  Rd[B == BL], Gr[B == BL], Bl[B == BL] = 0.2, 0.7, 0.9
  
  Rd[B == ABSORBING], Gr[B == ABSORBING], Bl[B == ABSORBING] = 0.5, 0.5, 0.5
  
  RGB[:,:,0], RGB[:,:,1], RGB[:,:,2] = Rd, Gr, Bl
  
  print(np.max(RGB), np.min(RGB))
  if bInvertBG: RGB[np.where((RGB==[0,0,0]).all(axis=2))] = [255,255,255]
  return RGB 


def squareSim(n=100, BC='PERIODIC'):
  B = np.zeros( (n,n) ).astype(int)
  first, last = 0, n-1  

  if BC == 'DIRICHLET': #-- DIRICHLET BOUNDARIES --
    B[:, first], B[ : ,last]  = DIRICHLET_LEFT, DIRICHLET_RIGHT
    B[first, :] += DIRICHLET_TOP 
    B[last,  :] += DIRICHLET_BOTTOM
  elif BC == 'NEUMANN': #-- NEUMANN BOUNDARIES --
    B[ : , first], B[ : , last]  = REFLECT_LEFT, REFLECT_RIGHT
    B[first,  :] += REFLECT_TOP 
    B[last,   :] += REFLECT_BOTTOM
  elif BC == 'PERIODIC': #-- PERIODIC BOUNDARIES --
    B[: ,first], B[:, last], B[first, :], B[last, :] = PERIODIC, PERIODIC, PERIODIC, PERIODIC 
  elif BC == 'ABSORBING':
    B[:, first], B[:, last], B[first, :], B[last, :] = ABSORBING, ABSORBING, ABSORBING, ABSORBING

  print('squareSim', n, BC)
  return B 
  
  
def processBcIni(lines):
  sz = lines[0].split(',')
  B = np.zeros( (int(sz[0]), int(sz[1])) ).astype(int)
  for line in lines[1:len(lines)]:
     if len(line):
         if line[0] != '#': exec(line)
  return B           


def grid2rgb(U):
  B = (U == 9999.99)
  #print(len(B)) 
  RGB = np.zeros( (U.shape[0], U.shape[1], 3) )
  Rd,Gr,Bl = copy.copy(U), np.zeros(U.shape), copy.copy(U)
  Rd[U < 0] = 0
  Bl[U > 0] = 0
  Gr[B] = 1
  Rd[B] = 0
  Bl[B] = 0
  
  RGB[:,:,0] = Rd
  RGB[:,:,1] = Gr
  RGB[:,:,2] = np.abs(Bl)
   
  mx = np.average(RGB)
  return RGB/mx

        
if __name__ == '__main__':
  cmds = sys.argv[1:len(sys.argv)]
  if len(cmds): fname = cmds[0]
  else:         fname = 'BC.one.ini'
  f = open(fname, 'r')
  lines = f.readlines()
  f.close()
  B = processBcIni(lines)
  RGB = boundary2rgb(B)

  fig, axes = plt.subplots(figsize=(10, 10))
  plt.imshow(RGB)
  plt.savefig('IO/'+fname.replace('.','_')+'.png', dpi=300, bbox_inches='tight')
  plt.clf()
