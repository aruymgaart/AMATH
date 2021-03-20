from tensorFiles import *
from videoFunctions import *

images = tnsrFile2numpy('data/ski_drop.npz') 
ret = []
for im in images:
	print(im.shape)
	im = im[150:800,0:500]
	print(im.shape)
	ret.append(im)
ret = np.array(ret)
np2movieFile(ret/np.max(ret), "data/ski_drop_c", invert=True)
numpy2tnsrFile(ret, "data/ski_drop_c.npz", dataType='float32')