import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def adjustFigAspect(fig,aspect=1):
	xsize,ysize = fig.get_size_inches()
	minsize = min(xsize,ysize)
	xlim = .4*minsize/xsize
	ylim = .4*minsize/ysize
	if aspect < 1: xlim *= aspect
	else:          ylim /= aspect
	fig.subplots_adjust(left=.5-xlim, right=.5+xlim, bottom=.5-ylim, top=.5+ylim)
	
def rmWhitespace(axisOff=False):
	if axisOff: 
		plt.gca().set_axis_off()
		plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
	else:
		plt.subplots_adjust(hspace = 0, wspace = 0)	
	
	if axisOff: 
		plt.gca().xaxis.set_major_locator(plt.NullLocator())
		plt.gca().yaxis.set_major_locator(plt.NullLocator())
	plt.margins(0,0)
		
def plotSV(S, fname='', ttl=''):
	fig, ax = plt.subplots()
	if ttl != '' : plt.title(ttl)
	if False: ax.bar(np.arange(len(S)), np.log(S))
	else: ax.bar(np.arange(len(S)), S, color=(0.1, 0.2, 0.3, 1.0), width=0.5)
	xlabs, xticks = [], []
	if len(S) < 50: 
		xticks = np.arange(len(S))
		for j in range(len(S)): xlabs.append('PC%d' % (j+1))
	else:           
		for j in range(len(S)): 
			if j % 10 == 0.0:
				xlabs.append('%d' % (j+1))	
				xticks.append(j)
	ax.set_xticks(xticks)
	ax.set_xticklabels(xlabs)
	if fname == '': plt.show()
	else: 
		plt.savefig(fname, figsize=(8, 6), dpi=300, bbox_inches='tight')
		plt.clf()
		
def plotImageMatrix(IMG, nrow, ncol, fname='', ttl=''):
	rmWhitespace(axisOff=True)
	fig, ax = plt.subplots(nrow, ncol)
	if ttl != '' : fig.suptitle(ttl)
	for j in range(nrow):
		for i in range(ncol):
			indx = j*nrow + i
			ax[j,i].axis('off')
			ax[j,i].imshow(IMG[indx].reshape(28,28))
	if fname == '': plt.show()
	else: 
		plt.savefig(fname, figsize=(8, 6), dpi=300, bbox_inches='tight')
		plt.clf()
		
def plotFrame(mv, fr, fname):
	fig, ax = plt.subplots()
	ax.axis('off')
	ax.imshow(mv[fr])
	if fname == '': plt.show()
	else: 
		plt.savefig(fname, figsize=(8, 6), dpi=300, bbox_inches='tight')
		plt.clf()		

