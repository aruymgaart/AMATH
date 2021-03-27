#################################################
# AP Ruymgaart 1/15/2019
#################################################

import os

try: import numpy as np
except: np = None

try: import pandas as pd
except: pd = None

try: import rpy2.robjects as ro
except: ro = None

try: import json as jn
except: jn = None

try: from skimage import color
except: color = None

try: from scipy import ndimage, misc
except: ndimage = None

try: import imageio
except: imageio = None

try: import P3_in_mem_zip as mz
except: mz = None

tnsrDataTypes = ["float64","float32","float16","int32","int16","uint32","uint16","byte"]


def parseLongCsvString(lstr):

	nmbrs = []
	sVal = ""

	for c in lstr:
		
		if c == ',':
			nr = float(sVal)
			sVal = ""
			nmbrs.append(nr)

		else:
			sVal += c

	if len(sVal): 
		nr = float(sVal)
		nmbrs.append(nr)
	print(len(nmbrs))	
	return nmbrs


def rfile2numpy(filename):

	szR = 'read.table(\"%s\")' % (filename)
	rM = ro.r(szR)
	return np.matrix(rM)


def rfile2r(filename):

	szR = 'read.table(\"%s\")' % (filename)
	rM = ro.r(szR)
	return rM


def tnsrFile2numpy(fname):

	try:
		fname = os.path.expandvars(fname.strip())
	except:
		print(["could not OS-expand (ENV)", fname])
		return np.array([])

	if fname.find(".npz") > -1:
		try:	
			testrec = np.load(fname)
			for key in testrec:
				return testrec[key] #- return 1st
		except:
			print(["could not find .npz", fname])
			return np.array([])

	return np.array([])


def numpy2tnsrFile(ndarr, fname, fmt="npz", z=True, dataType=None, name=""):

	shape = ndarr.shape
	
	if fmt == 'npz':
		if not dataType:
			np.savez_compressed(fname, ndarr)
		else:
			np.savez_compressed(fname, ndarr.astype(dataType))
		return
	else: return -1
	