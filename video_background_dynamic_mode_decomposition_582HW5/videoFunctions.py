import imageio, cv2, numpy as np
from skimage.color import rgb2grey
from skimage.transform import resize
from tensorFiles import *

def video2numpy(fname):
	imgs,img = [], 1
	cap = cv2.VideoCapture(fname)
	while not img is None:
		ret, img = cap.read()
		if not img is None:	imgs.append(img)
	return np.array(imgs)

def flattenVideo(npVid, strt=0, stp=None):
	szX, szY, X = npVid[0].shape[1], npVid[0].shape[0], []
	for im in npVid: X.append(im.reshape(-1))
	X = np.array(X)
	f1, f2 = strt, len(X)
	if not stp is None: f2 = stp
	return X[f1:f2], szX, szY
	
def reshape2video(flatImgs, szY, szX):
	npMv = []
	for im in flatImgs: 
		npMv.append(np.abs(im.reshape(szY, szX)))
	return np.array(npMv)		

def halfGreyscale(imgs):
	ret = []
	szy, szx = int(imgs[0].shape[0]/2), int(imgs[0].shape[1]/2)
	for img in imgs: ret.append(resize(rgb2grey(img), (szy, szx)))
	return np.array(ret)
	
def np2movieFile(imgTensor, fname, fps=30, invert=True):
	if invert: imgTensor = 1.0 - imgTensor
	imgTensor = (imgTensor *255.0).astype('uint8')
	imageio.mimwrite(fname+'.mp4', imgTensor , fps = fps)
	
def processCmdArgs(arglst):
	cmd = {}
	for c in arglst:
		try:
			elms = c.split('=')
			cmd[elms[0]] = elms[1]
		except: pass
	return cmd		
	
if __name__ == "__main__":
	mcVideo = video2numpy("data/monte_carlo.mov")
	mcVideo = halfGreyscale(mcVideo)
	numpy2tnsrFile(mcVideo, "data/monte_carlo.npz", dataType='float32')
	skiVideo = video2numpy("data/ski_drop.mov")
	skiVideo = halfGreyscale(skiVideo) 
	numpy2tnsrFile(skiVideo, "data/ski_drop.npz", dataType='float32')	

	