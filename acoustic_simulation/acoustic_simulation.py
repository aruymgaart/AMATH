# AP Ruymgaart
# Finite difference 2D wave equation 
import copy, sys, numpy as np, matplotlib.pyplot as plt
from draw_boundaries import *
from tensorFiles import *
from pde_findiff_functions import *
fft2, fftshift, ifft2 = np.fft.fft2, np.fft.fftshift, np.fft.ifft2

def waveEquationUnext(U,Ulast,S,B,a1,a2):
  Un = np.zeros(U.shape)   	
  for j in range(U.shape[0]):
    for i in range(U.shape[1]):
      u_pm, u_pc, u_pp, u_cm, u_cc, u_cp, u_mm, u_mc, u_pp = stencil(j,i,U,B)
      ux = u_cm - 2*u_cc + u_cp
      uy = u_pc - 2*u_cc + u_mc
      Un[j,i] = a1*ux + a2*uy + 2*u_cc - Ulast[j,i] + S[j,i]
  return Un

def waveEquationSpectralUNext(Uft, Ulast, dt, c, Lft):
  fact = (c*dt)**2
  UnFt = fact*np.multiply(Uft,Lft) + 2*Uft - Ulast
  return UnFt 

def source(t):
  S = np.zeros( (200,180) )
  v = np.cos(t)
  S[121:175,19] = v
  S[121:175,20] = -v
  return S

def guassian(r, x, y, px, py):          return np.exp(-r*(x-px)**2 - r*(y-py)**2  )
def gaussianSource(t, r, x, y, px, py): return np.cos(t)*guassian(r, x, y, px, py)  

if __name__ == '__main__':
  IC, source, name = 'none', 'none', 'none'
  simsel, nrSteps, c, dt, W, mod = 1, 1000, 343.0, 0.000001, 0.2, 100 
  plot, bSpectral = False, False
  
  cmds = sys.argv[1:len(sys.argv)]
  if len(cmds) > 0 :
    stype = cmds[0].split(':') 
    simsel = 2
    if stype[0] == 'SQUARE':
      n = int(stype[1])
      B   = squareSim(n, stype[2]) # Boundary matrix 
      name = 'square_%s' % (stype[2])
    elif stype[0] == 'FILE':
      f = open(stype[1], 'r')
      lines = f.readlines()
      f.close()
      B =  processBcIni(lines) 
      name = stype[1].replace('.ini','').replace('BC.','')  

  if len(cmds) > 1 : nrSteps = int(cmds[1])
  if len(cmds) > 2 : dt = float(cmds[2])
  if len(cmds) > 3 : plot = cmds[3].lower()[0] == 't'
  if len(cmds) > 4 : mod = int(cmds[4])
  if len(cmds) > 5 : IC = cmds[5].split(':')
  if len(cmds) > 6 : source = cmds[6].split(':')
  if len(cmds) > 7 : 
    if cmds[7] == 'SPECTRAL' : 
      bSpectral = True
      name += '_SPECTRAL'
  if len(cmds) > 8 : W = float(cmds[8])

  ds  = W/B.shape[1]   # delta_space (x and y)
  H   = ds*B.shape[0]
  CFL =(c*dt)/ds
  a1, a2 = CFL**2, CFL**2
  stable = abs(CFL) < 1/np.sqrt(2)
  bcFact = (CFL - 1)/(CFL + 1)
  print('Time step', dt, 'width=', W,'height=', H, 'nr steps', nrSteps, 'Sim=', simsel, 'IC', IC, 'Source', source)
  print('CFL=', CFL, 'a1', a1, 'STABLE', stable, 'Space ds=', ds)
  if not stable: exit()
  if bSpectral: 
    if not stype[0] == 'SQUARE':
      print('not supported') 
      exit()

  vU = np.zeros( (nrSteps, B.shape[0], B.shape[1]) )
  if bSpectral: vU = vU.astype(complex)

  x2 = np.linspace(0, W, B.shape[1]+1)
  y2 = np.linspace(0, H, B.shape[0]+1)
  x, y = x2[0:B.shape[1]], y2[0:B.shape[0]]
  [X,Y] = np.meshgrid(x,y)
  n = B.shape[0]
  k = (2*np.pi/(H)) * np.append(np.arange(0,n/2),np.arange(-n/2,0)) 
  [Kx,Ky] = np.meshgrid(k,k)
  Kx2, Ky2 = np.multiply(Kx,Kx), np.multiply(Ky,Ky)
  Lft = -1.0*(Kx2 + Ky2) #-- Spectral Laplacian --
 
  if IC[0] == 'GAUSS':
    G = guassian(40000, X, Y, W*float(IC[1]), H*float(IC[2]))
    if bSpectral:
      vU[0,:,:], vU[1,:,:] = fft2(G), fft2(G)
    else:
      vU[0,:,:], vU[1,:,:] = G, G
  elif IC[0] == 'RECT':
    ix1, ix2, iy1, iy2 = int(IC[1]), int(IC[2]), int(IC[3]), int(IC[4])
    R = np.zeros(B.shape)
    R[iy1:iy2,ix1:ix2] = float(IC[5])
    R[B != 0] = 0.0
    vU[0,:,:], vU[1,:,:] = R, R

  if source[0] == 'GAUSS':
    b = float(source[3])*np.pi 
    GS = guassian(40000, X, Y, W*float(source[1]), H*float(source[2]))
   
  for k in range(2,nrSteps):   
    if source[0] == 'GAUSS':
      S = GS*np.cos(k*b) # gaussianSource(k*b, 40000, X, Y, W*float(source[1]), H*float(source[2]))
    elif simsel == 7:
      b = 0.09*np.pi  # 0.0009 seems to show resonance at dt=0.0006   
      S = source(k*b) 
    else:
      S = np.zeros(B.shape)

    Ulast = vU[k-2]                                # time i-1
    U = vU[k-1]                                    # time i
    if bSpectral:
      Un = waveEquationSpectralUNext(U, Ulast, dt, c, Lft)
    else:
      Un = waveEquationUnext(U, Ulast, S, B, a1, a2) # this is time i+1

    # special case absorbing boundary available at edges only
    # dealt with here rather than in stencil
    jEnd = B.shape[0] - 1
    iEnd = B.shape[1] - 1
    for j in range(U.shape[0]):
      if B[j,0] == ABSORBING:
        Un[j,0] = U[j,1]  + bcFact*(Un[j,1] - U[j,0] )
      if B[j,iEnd] == ABSORBING:
        Un[j,iEnd] = U[j,iEnd-1]  + bcFact*(Un[j,iEnd-1] - U[j,iEnd] )

    for i in range(U.shape[1]):
      if B[0,i] == ABSORBING:
        Un[0,i] = U[1,i] + bcFact*(Un[1,i] -  U[0,i])
      if B[jEnd,i] == ABSORBING:
        Un[jEnd,i] = U[jEnd-1,i] + bcFact*(Un[jEnd-1,i] -  U[jEnd,i]) 

    vU[k] = Un
              
    if k % mod == 0.0:
      print("step", k)      
      if plot:
        if bSpectral: 
          img = np.real(ifft2(vU[k]))
          img = np.average(img) - img  # ?
        else:
          img = vU[k]
        im = grid2rgb(img)
        im[B != 0] = 1
        fig, axes = plt.subplots(figsize=(8, 8))
        plt.imshow(im)
        if False:
          plt.show()
        else:
          pltname = 'IO/sim_%s_%d_%s_%s.png' % (name, k, cmds[5].replace(':','-').replace('0.',''), 
                                                    cmds[6].replace(':','-').replace('0.','') )
          plt.savefig(pltname, dpi=300, bbox_inches='tight')
          plt.clf()
        
  if bSpectral: 
    realU = np.zeros( (nrSteps, B.shape[0], B.shape[1]) )
    for k in range(nrSteps): 
      img = np.real(ifft2(vU[k]))
      realU[k] = np.average(img) - img        
    vU = realU 

  for k in range(2,nrSteps): vU[k][B != 0] = 9999.99    
  numpy2tnsrFile(vU, 'sim_%s_%d_%s_%s.npz' % (name, nrSteps, cmds[5].replace(':','-').replace('0.',''), 
                                                cmds[6].replace(':','-').replace('0.','')) )        
  










