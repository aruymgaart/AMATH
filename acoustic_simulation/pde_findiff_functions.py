# AP Ruymgaart
# Finite difference PDE solver
# STENCIL FUNCTION 
import copy, sys, numpy as np 

PERIODIC                                                         =   1               #
REFLECT_TOP, REFLECT_BOTTOM, REFLECT_LEFT, REFLECT_RIGHT         =   2,  4,   8,  16 # green, red, blue, yellow
DIRICHLET_TOP, DIRICHLET_BOTTOM, DIRICHLET_LEFT, DIRICHLET_RIGHT =  32, 64, 128, 256 #
ABSORBING                                                        = 512               # black
bcNames = {  1:'PERIODIC',          2: 'REFLECT_TOP',    4:'REFLECT_BOTTOM', 
             8:'REFLECT_LEFT',     16:'REFLECT_RIGHT',  32:'DIRICHLET_TOP',
            64:'DIRICHLET_BOTTOM',128:'DIRICHLET_LEFT',256:'DIRICHLET_RIGHT',
		   512:'ABSORBING'}
'''
STENCIL FUNCTION
returns 3x3 grid while applying any combination of boundary conditions as specified in boundary matrix B
  u_pm, u_pc, u_pp
  u_cm, u_cc, u_cp
  u_mm, u_mc, u_pp 
'''
def stencil(j,i,G,B,d=0, verbose=False):
  Ly, Lx = B.shape[0], B.shape[1]
  bc = B[j,i]
  ym, yc, yp, xm, xc, xp = j-1, j, j+1, i-1, i, i+1
  if verbose and bc !=0:
        szName = "\n"
        for bit in [1,2,4,8,16,32,64,128,256,512]: 
           if bc & bit: szName += "%s " % ( bcNames[bit] )         
        print(szName, ym, yc, yp, xm, xc, xp)   

  #-- ABSORBING -- 
  if bc == ABSORBING: return 0,0,0,0,0,0,0,0,0 # defer (put zeros for now)
  
  #-- INTERIOR (NO BC) --
  if bc == 0:
    u_pm, u_pc, u_pp = G[yp,xm], G[yp,xc], G[yp,xp]
    u_cm, u_cc, u_cp = G[yc,xm], G[yc,xc], G[yc,xp]
    u_mm, u_mc, u_pp = G[ym,xm], G[ym,xc], G[ym,xp]
    return u_pm, u_pc, u_pp, u_cm, u_cc, u_cp, u_mm, u_mc, u_pp       
 
  #-- PERIODIC (at rectangular box edges only) --
  if bc == PERIODIC:
    if ym < 0   : ym += Ly 
    if yp >= Ly : yp -= Ly
    if xm < 0   : xm += Lx 
    if xp >= Lx : xp -= Lx
    u_pm, u_pc, u_pp = G[yp,xm], G[yp,xc], G[yp,xp]
    u_cm, u_cc, u_cp = G[yc,xm], G[yc,xc], G[yc,xp]
    u_mm, u_mc, u_pp = G[ym,xm], G[ym,xc], G[ym,xp]
    return u_pm, u_pc, u_pp, u_cm, u_cc, u_cp, u_mm, u_mc, u_pp 
            
  bNeumann = bc & (REFLECT_TOP + REFLECT_BOTTOM + REFLECT_LEFT + REFLECT_RIGHT) !=0           

  #-- DIRICHLET (as written, at edges only) --
  u_pm, u_pc, u_pp, u_cm, u_cc, u_cp, u_mm, u_mc, u_pp = d,d,d,d,d,d,d,d,d
  if not bNeumann:
    if bc & DIRICHLET_BOTTOM:
      if bc & DIRICHLET_LEFT:
        u_cm, u_cc, u_cp = d, G[yc,xc], G[yc,xp]
        u_mm, u_mc, u_pp = d, G[ym,xc], G[ym,xp]    
      elif bc & DIRICHLET_RIGHT:
        u_cm, u_cc, u_cp = G[yc,xm], G[yc,xc], d
        u_mm, u_mc, u_pp = G[ym,xm], G[ym,xc], d    
      else: 
        u_cm, u_cc, u_cp = G[yc,xm], G[yc,xc], G[yc,xp]
        u_mm, u_mc, u_pp = G[ym,xm], G[ym,xc], G[ym,xp]    
    elif bc & DIRICHLET_TOP:
      if bc & DIRICHLET_LEFT:
        u_pm, u_pc, u_pp = d, G[yp,xc], G[yp,xp]
        u_cm, u_cc, u_cp = d, G[yc,xc], G[yc,xp]  
      elif bc & DIRICHLET_RIGHT:
        u_pm, u_pc, u_pp = G[yp,xm], G[yp,xc], d
        u_cm, u_cc, u_cp = G[yc,xm], G[yc,xc], d  
      else:  
        u_pm, u_pc, u_pp = G[yp,xm], G[yp,xc], G[yp,xp]
        u_cm, u_cc, u_cp = G[yc,xm], G[yc,xc], G[yc,xp]       
    else:        
      if bc & DIRICHLET_LEFT:
        u_pm, u_pc, u_pp = d, G[yp,xc], G[yp,xp]
        u_cm, u_cc, u_cp = d, G[yc,xc], G[yc,xp]
        u_mm, u_mc, u_pp = d, G[ym,xc], G[ym,xp]   
      elif bc & DIRICHLET_RIGHT:
        u_pm, u_pc, u_pp = G[yp,xm], G[yp,xc], d
        u_cm, u_cc, u_cp = G[yc,xm], G[yc,xc], d
        u_mm, u_mc, u_pp = G[ym,xm], G[ym,xc], d        
    return u_pm, u_pc, u_pp, u_cm, u_cc, u_cp, u_mm, u_mc, u_pp             

  #-- NEUMANN --
  if bc & REFLECT_TOP: 
    ym, yc, yp = j+1, j, j+1
  if bc & REFLECT_BOTTOM: 
    ym, yc, yp = j-1, j, j-1
  if bc & REFLECT_LEFT:
    xm, xc, xp = i+1, i, i+1
  if bc & REFLECT_RIGHT:
    xm, xc, xp = i-1, i, i-1
  try:
    u_pm, u_pc, u_pp = G[yp,xm], G[yp,xc], G[yp,xp]
    u_cm, u_cc, u_cp = G[yc,xm], G[yc,xc], G[yc,xp]
    u_mm, u_mc, u_pp = G[ym,xm], G[ym,xc], G[ym,xp]
  except:
    szName = ""  
    for bit in [1,2,4,8,16,32,128]:
       if bc & bit: szName += "%s " % ( bcNames[bit] )     
    print('BC ERROR', szName, 'at', j,i, 'rel pos y',  ym, yc, yp, 'rel pos x', xm, xc, xp)
    exit()
  return u_pm, u_pc, u_pp, u_cm, u_cc, u_cp, u_mm, u_mc, u_pp

