import numpy as np
from os.path import dirname, join as pjoin
import scipy.io as sio
from skimage.color import rgb2grey
import scipy.misc
from skimage.transform import resize
from scipy.signal import convolve2d
import pickle

#============ camera 1 ==============
mat_contents = sio.loadmat('data/cam1_1.mat')
images1 = mat_contents['vidFrames1_1']
mat_contents = sio.loadmat('data/cam1_2.mat')
images2 = mat_contents['vidFrames1_2']
mat_contents = sio.loadmat('data/cam1_3.mat')
images3 = mat_contents['vidFrames1_3']
mat_contents = sio.loadmat('data/cam1_4.mat')
images4 = mat_contents['vidFrames1_4']
print(mat_contents.keys())
print('CAM1-1:', images1.shape)
print('CAM1-2:', images2.shape)
print('CAM1-3:', images3.shape)
print('CAM1-4:', images4.shape)

M, szX, szY = [], None, None
for images in [images1, images2, images3, images4]:
	T = []
	for i in range(images.shape[3]):
		if False: im = resize(rgb2grey(images[:,:,:,i]), (240,320) )
		else: im = resize(images[:,:,:,i], (240,320) )
		im = im[100:220,100:250]
		if szX is None: szX, szY = im.shape[1], im.shape[0]
		T.append(im)		
	M.append(np.array(T))

f = open('data/camera_1.dict', "wb")		
pickle.dump(M,f)
f.close()

#============ camera 1 ==============
mat_contents = sio.loadmat('data/cam2_1.mat')
images1 = mat_contents['vidFrames2_1']
mat_contents = sio.loadmat('data/cam2_2.mat')
images2 = mat_contents['vidFrames2_2']
mat_contents = sio.loadmat('data/cam2_3.mat')
images3 = mat_contents['vidFrames2_3']
mat_contents = sio.loadmat('data/cam2_4.mat')
images4 = mat_contents['vidFrames2_4']
print('CAM2-1:', images1.shape)
print('CAM2-2:', images2.shape)
print('CAM2-3:', images3.shape)
print('CAM2-4:', images4.shape)	

M, szX, szY = [], None, None
for images in [images1, images2, images3, images4]:
	T = []
	for i in range(images.shape[3]):
		if False: im = resize(rgb2grey(images[:,:,:,i]), (240,320) )
		else: im = resize(images[:,:,:,i], (240,320) )
		im = im[25:215,80:250]
		if szX is None: szX, szY = im.shape[1], im.shape[0]
		T.append(im)		
	M.append(np.array(T))
f = open('data/camera_2.dict', "wb")		
pickle.dump(M,f)
f.close()

#============ camera 3 ==============
mat_contents = sio.loadmat('data/cam3_1.mat')
images1 = mat_contents['vidFrames3_1']
mat_contents = sio.loadmat('data/cam3_2.mat')
images2 = mat_contents['vidFrames3_2']
mat_contents = sio.loadmat('data/cam3_3.mat')
images3 = mat_contents['vidFrames3_3']
mat_contents = sio.loadmat('data/cam3_4.mat')
images4 = mat_contents['vidFrames3_4']	
print('CAM3-1:', images1.shape)
print('CAM3-2:', images2.shape)
print('CAM3-3:', images3.shape)
print('CAM3-4:', images4.shape)

M, szX, szY = [], None, None
for images in [images1, images2, images3, images4]:
	T = []
	for i in range(images.shape[3]):
		if False: im = resize(rgb2grey(images[:,:,:,i]), (240,320) )
		else: im = resize(images[:,:,:,i], (240,320) )
		im = im[45:215,80:280]
		if szX is None: szX, szY = im.shape[1], im.shape[0]
		T.append(im)		
	M.append(np.array(T))
f = open('data/camera_3.dict', "wb")		
pickle.dump(M,f)
f.close()
