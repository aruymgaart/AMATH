import numpy as np

def stationaryPoints(x, bUseSign=False, tol=0.01): 
	dxdt = np.gradient(x)
	if bUseSign:
		asign = np.sign(dxdt)
		sc = ((np.roll(asign, 1) - asign) != 0).astype(int)
		return np.argwhere(np.array(sc) == 1).reshape(-1)
	else: return np.where(abs(dxdt) < tol)[0]

def removeLeadingStationary(x):
	dxdt = np.gradient(x)
	p, z = 0, dxdt[0]
	while z == 0:
		z = dxdt[p] 
		p += 1
	return x[p:len(x)], p

def firstMinimum(sPoints, x):
	dxdt = np.gradient(x)
	dx2dt2 = np.gradient(dxdt)
	for p in sPoints:
		if dx2dt2[p] > 0 : return p
	return None

def firstMaximum(sPoints, x):
	dxdt = np.gradient(x)
	dx2dt2 = np.gradient(dxdt)
	for p in sPoints:
		if dx2dt2[p] < 0 : return p
	return None

