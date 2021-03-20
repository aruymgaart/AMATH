# AP Ruymgaart.  This script reads the pickled pre-processed movies and produces object coordinates from each
import numpy as np
from scipy.signal import convolve2d
from skimage.color import rgb2grey
import matplotlib.pyplot as plt
import pickle, copy
from imageFunctions import *

M1,M2,M3 = pickle.load(open('data/camera_1.dict','rb')), pickle.load(open('data/camera_2.dict','rb')), pickle.load(open('data/camera_3.dict','rb'))

canM1 = M1[0][29][67:114:,58:89,:]
canM2 = M2[0][0][108:155:,50:84,:]
canM3 = M3[0][150][86:115,121:159,:]

imageSetToObjectCoordinates( [M1[0]], [canM1], name='data/cam1_1' )
imageSetToObjectCoordinates( [M1[1]], [canM1], name='data/cam1_2' )		
imageSetToObjectCoordinates( [M2[0]], [canM2], name='data/cam2_1' )
imageSetToObjectCoordinates( [M2[1]], [canM2], name='data/cam2_2' )
imageSetToObjectCoordinates( [M3[0]], [canM3], name='data/cam3_1' )
imageSetToObjectCoordinates( [M3[1]], [canM3], name='data/cam3_2' )
imageSetToObjectCoordinates( [M1[2]], [canM1], name='data/cam1_3' )

w,h,s = 20,26, 2
can1 = getCanImg(M2,s,170,71,129,w,h,offy=8) 
can2 = getCanImg(M2,s,196,92,127,w,h,offy=0)
can3 = getCanImg(M2,s,203,99,134,w,h,offy=12)
obs = [canM2,can1,can2,can3] #- perfect
imageSetToObjectCoordinates( [ M2[2] ], obs, name='data/cam2_3' )

w,h,s = 20,16, 2
can0,can1 = getCanImg(M3,s,0,113,75,w,h,offy=0),  getCanImg(M3,s,14,115,105,w,h,offy=0)
can2,can3 = getCanImg(M3,s,52,127,71,w,h,offy=0), getCanImg(M3,s,72,84,87,w,h,offy=0)
obs = [can0,can1,can2,can3] #- perfect
imageSetToObjectCoordinates( [ M3[2] ], obs, name='data/cam3_3' )

#== CANS AT 16 ANGLES (guessed about 360 degrees/16) ==
w,h,s = 14,17, 3
can1, can2 =  getCanImg(M1,s,0,98,46,w,h),  getCanImg(M1,s,10,104,41,w,h)
can3, can4 =  getCanImg(M1,s,16,110,44,w,h),getCanImg(M1,s,19,110,50,w,h)
can5, can6 =  getCanImg(M1,s,22,110,60,w,h),getCanImg(M1,s,26,106,69,w,h)
can7, can8 =  getCanImg(M1,s,31,101,76,w,h),getCanImg(M1,s,35,93,76,w,h)
can9, can10 = getCanImg(M1,s,44,82,49,w,h), getCanImg(M1,s,50,81,38,w,h)
can11,can12 = getCanImg(M1,s,55,80,40,w,h), getCanImg(M1,s,64,82,65,w,h)
can13,can14 = getCanImg(M1,s,74,90,80,w,h), getCanImg(M1,s,79,92,70,w,h)
can15,can16 = getCanImg(M1,s,94,92,41,w,h), getCanImg(M1,s,111,88,73,w,h)

obs = [can1,can2,can3,can4,can5,can6,can7,can8,can9,can10,can11,can12,can13,can14,can15,can16]
imageSetToObjectCoordinates( [ M1[3] ], obs, name='data/cam1_4')

if False: #- alternate way
	imageSetToObjectCoordinates([M1[1]], rgb2grey(M1[0][29][67:114:,58:89,:]), bConv=True, name='data/cam1_4')
	#imageSetToObjectCoordinates([M1[3]], M1[0][29][67:114:,58:89,:], bConv=False, name='cam1_4')

#== CANS AT 16 ANGLES (guessed about 360 degrees/16) ==
w,h,s, oy = 17,20, 3, 10
can1, can2 =  getCanImg(M2,s, 0, 48,108,w,h,offy=oy), getCanImg(M2,s, 5, 61, 93,w,h,offy=oy) 
can3, can4 =  getCanImg(M2,s,12, 81, 69,w,h,offy=oy), getCanImg(M2,s,17, 96, 58,w,h,offy=oy)
can5, can6 =  getCanImg(M2,s,20,104, 59,w,h,offy=oy), getCanImg(M2,s,24,104, 68,w,h,offy=oy)
can7, can8 =  getCanImg(M2,s,29,102, 90,w,h,offy=oy), getCanImg(M2,s,33, 94,105,w,h,offy=oy)
can9, can10 = getCanImg(M2,s,37, 86,114,w,h,offy=oy), getCanImg(M2,s,42, 74,110,w,h,offy=oy)
can11,can12 = getCanImg(M2,s,46, 65, 98,w,h,offy=oy), getCanImg(M2,s,49, 59, 82,w,h,offy=oy)
can13,can14 = getCanImg(M2,s,53, 57, 65,w,h,offy=oy), getCanImg(M2,s,58, 55, 52,w,h,offy=oy)
can15,can16 = getCanImg(M2,s,67, 64, 73,w,h,offy=oy), getCanImg(M2,s,74, 75,107,w,h,offy=oy)
obs = [can1,can2,can3,can4,can5,can6,can7,can8,can9,can10,can11,can12,can13,can14,can15,can16]
plotCanRot(can1,can2,can3,can4,can5,can6,can7,can8,can9,can10,can11,can12,can13,can14,can15,can16,fname='images/canR16_2_4.png')
imageSetToObjectCoordinates( [ M2[3] ], obs, name='data/cam2_4' )

w,h,s =18,15,3
can1, can2 = getCanImg(M3,s,  0,113,63,w,h), getCanImg(M3,s,  4,102,61,w,h)
can3, can4 = getCanImg(M3,s,  7, 95,60,w,h), getCanImg(M3,s, 11, 93,58,w,h)
can5, can6 = getCanImg(M3,s, 16,100,56,w,h), getCanImg(M3,s, 20,107,54,w,h)
can7, can8 = getCanImg(M3,s, 23,117,52,w,h), getCanImg(M3,s, 30,134,53,w,h)
can9,can10 = getCanImg(M3,s, 35,132,59,w,h), getCanImg(M3,s, 41,116,65,w,h)
can11,can12= getCanImg(M3,s, 51, 98,67,w,h), getCanImg(M3,s, 61,113,57,w,h)
can13,can14= getCanImg(M3,s, 73,134,44,w,h),getCanImg(M3,s, 80,118,51,w,h)
can15,can16= getCanImg(M3,s, 85,105,60,w,h),getCanImg(M3,s, 92,104,63,w,h)

obs = [can1,can2,can3,can4,can5,can6,can7,can8,can9,can10,can11,can12,can13,can14,can15,can16]
plotCanRot(can1,can2,can3,can4,can5,can6,can7,can8,can9,can10,can11,can12,can13,can14,can15,can16,fname='images/canR16_3_4.png')
imageSetToObjectCoordinates( [M3[3]], obs, name='data/cam3_4' )

