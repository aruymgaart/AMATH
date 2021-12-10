#######################################################################
# Simulation Tensor (Movie) Viewer
# AP Ruymgaart 5/25/2018
#######################################################################
#
#######################################################################
import time, sys, os
import random
import numpy as np
import scipy
import copy
from math import factorial, log
from os import listdir
from os.path import isfile, join

import PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import (Qt, pyqtSignal)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from tensorFiles import *
from draw_boundaries import grid2rgb

#######################################################################
class ImagePanel(QtWidgets.QWidget):
	
	imgPnlCallback = pyqtSignal()
	imageMouseClick = pyqtSignal()
	
	#----------------------------------------------------------------------
	def __init__(self, parent = None):

		super(ImagePanel, self).__init__()


		self.image = np.random.randn(300,300,3)

		self.figure = plt.figure()
		self.canvas = FigureCanvas(self.figure)
		self.mouseX = 0
		self.mouseY = 0

		cid = self.canvas.mpl_connect('button_press_event', self.onclick) #fig.canvas

		self.MainSizer = QtWidgets.QVBoxLayout()
		self.LineSizer = QtWidgets.QHBoxLayout()

		self.MainSizer.addLayout(self.LineSizer)

		self.MainSizer.addWidget(self.canvas)
		self.setLayout(self.MainSizer)


	def onclick(self, event):
		print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
		('double' if event.dblclick else 'single', event.button, event.x, event.y, event.xdata, event.ydata))

		self.mouseX = event.xdata
		self.mouseY = event.ydata

		self.imageMouseClick.emit()


	def clearPlot(self):

		plt.gcf().clear()
		self.canvas.draw()


	def plotImage(self, img, showTicks=False, thresh=8.0):

		self.image = grid2rgb(img)
		#self.image[self.image == 1] = 255
		
		if False: 
			print('plotImage', img.shape)
			if len(img.shape) > 2:
				print('R', np.max(img[:,:,0]), np.min(img[:,:,0]))
				print('G', np.max(img[:,:,1]), np.min(img[:,:,1]))
				print('B', np.max(img[:,:,2]), np.min(img[:,:,2]))


		plt.gcf().clear()
		#plt.gca().set_axis_off()
		plt.margins(0,0)
		plt.autoscale(tight=True)

		ax = self.figure.add_subplot(111) #,figsize=(18, 2))#

		#figsize=(18, 2)
		
		if not showTicks:
			#ax.set_adjustable('box-forced')
			ax.get_xaxis().set_visible(False)
			ax.axes.get_yaxis().set_visible(False)
			#axis('scaled')
			plt.tight_layout()
		
		if thresh != 0.0:
			av = np.average(self.image)
			im = copy.copy(self.image)

			if False:
				print('thresh', im.shape, 'avg=', av)
				print('R', np.max(im[:,:,0]), np.min(im[:,:,0]))
				print('G', np.max(im[:,:,1]), np.min(im[:,:,1]))
				print('B', np.max(im[:,:,2]), np.min(im[:,:,2]))

			im[im > av*thresh] = av*thresh

			if False:
				print('R', np.max(im[:,:,0]), np.min(im[:,:,0]))
				print('G', np.max(im[:,:,1]), np.min(im[:,:,1]))
				print('B', np.max(im[:,:,2]), np.min(im[:,:,2]))

			mx = np.max(im)
			im = im * (255.0/mx)
			ax.imshow(im)			
		else:	
			ax.imshow(self.image)

		self.canvas.draw()		
	
		
#######################################################################
class imageMovieViewer(QtWidgets.QWidget):

	def __init__(self, parent = None):
		
		super(imageMovieViewer, self).__init__()
		
		self.tnsr = np.array((0,0,0))
		
		self.pnl = ImagePanel()
		self.n = 0
		self.run = False
		self.fwd = True
		self.stepTime = 1
		self.step = 1
		self.thresh = 0
		self.flipUD = False
		self.flipLR = False
		self.transpose = False
		
		self.sclast = 1
		self.app = None	
		
		#self.pProcessTensorFile = None	
		
		self.info = QtWidgets.QTextEdit()
		
		buttonLayout = QtWidgets.QHBoxLayout()
		
		self.btn1 = QtWidgets.QPushButton(self)
		self.btn2 = QtWidgets.QPushButton(self)
		self.btn3 = QtWidgets.QPushButton(self)
		self.btn1.setText('RUN')
		self.btn2.setText('FWD')
		self.btn3.setText('STEP')
		
		self.dropFileSelect = QtWidgets.QComboBox(self)	
		dpath = '.'
		onlyfiles = [f for f in listdir(dpath) if isfile(join(dpath, f))]
		tensorfiles = []
		for fname in onlyfiles:
			if len(fname) > 4:
				if fname[len(fname)-4:len(fname)] == '.npz': 
					tensorfiles.append(fname)
					self.dropFileSelect.addItem(fname)
		#print(tensorfiles)		
		self.dropFileSelect.currentIndexChanged.connect(self.fileSelected)

		self.speedSpin = QtWidgets.QSpinBox(self)
		self.speedSpin.setRange(1,10)
		self.speedSpin.setSingleStep(1)
		self.speedSpin.valueChanged.connect(self.speedChange)
		
		self.threshSpin = QtWidgets.QSpinBox(self)
		self.threshSpin.setRange(0,10)
		self.threshSpin.setSingleStep(1)
		self.threshSpin.valueChanged.connect(self.threshChange)		
		
		buttonLayout.addWidget(self.btn1)
		buttonLayout.addWidget(self.btn2)
		buttonLayout.addWidget(self.btn3)		
		buttonLayout.addWidget(self.speedSpin)
		buttonLayout.addWidget(self.threshSpin)

		v_splitter = QtWidgets.QSplitter(Qt.Vertical, self)						
	
		layout = QtWidgets.QVBoxLayout()
		layout2 = QtWidgets.QVBoxLayout()

		layout2.addWidget(self.dropFileSelect)
		layout2.addWidget(self.info)
		layout2.addLayout(buttonLayout)	
		bottomWidget = QtWidgets.QWidget()
		bottomWidget.setLayout(layout2)

		v_splitter.addWidget(self.pnl)
		v_splitter.addWidget(bottomWidget)

		layout.addWidget(v_splitter)			
		self.setLayout(layout)
		
		self.btn1.clicked.connect(self.btn1Callback)
		self.btn2.clicked.connect(self.btn2Callback)
		self.btn3.clicked.connect(self.btn3Callback)				
		
		self.timer = QtCore.QTimer()
		self.timer.timeout.connect(self.timerCallback)
		self.timer.start(200)
		
		
	def pProcessTensorFile(self, fname):
			
		return tnsrFile2numpy(fname)
		
	def fileSelected(self):
		
		fname = self.dropFileSelect.itemText(self.dropFileSelect.currentIndex() )
		print(fname)
		#tensor = mf.tnsrFile2numpy(fname)
		#print(tensor.shape)
		
		if self.pProcessTensorFile:
			self.tnsr = self.pProcessTensorFile(fname)
			print('LOADED TENSOR, shape=', self.tnsr.shape)
	
		
	def speedChange(self):
		
		self.stepTime = self.speedSpin.value()	
		
	def threshChange(self):
		
		self.thresh = self.threshSpin.value()			
		
	
	def plotFrame(self, flipUD=True, flipLR=True, transpose=False, fwd=True):
		
		#if not self.run: return
		#if not self.step % self.stepTime == 0.0: return
		print('plotFrame', self.n)
	
		im = self.tnsr[self.n]		
		
		ttl = np.sum(im)
		szInf = "frame %d \t %8.2f" % (self.n, ttl)
		self.info.setText(szInf)		
		
		print(self.n, im.shape, self.tnsr.shape)
		if transpose: im = im.T
		if flipUD: im = np.flipud(im)
		if flipLR: im = np.fliplr(im)
		self.pnl.plotImage(im, thresh=self.thresh)
		
		if fwd:
			self.n += 1
		else:	
			self.n -= 1
			
		if self.n >= len(self.tnsr): self.n = 0
		if self.n < 0 : self.n = len(self.tnsr) - 1 
		

		
		
	def timerCallback(self):	

		#print('timer', self.step, self.stepTime, self.step % self.stepTime )
		if self.run:
			if self.step % self.stepTime == 0.0:
				self.plotFrame(fwd=self.fwd, flipUD=self.flipUD, flipLR=self.flipLR, transpose=self.transpose)
		self.step += 1
		
		
	def btn1Callback(self):
		
		print('hi')
		self.run = not self.run
		
		
	def btn2Callback(self):
		
		print('hi')
		self.fwd = not self.fwd
		
		if self.fwd:
			self.btn2.setText('FWD')
		else:
			self.btn2.setText('BACK')
		self.myScreenWontUpdate()	
			
			
	def btn3Callback(self):
		
		print('btn3Callback, manual step')
		self.plotFrame(fwd=self.fwd)
		self.myScreenWontUpdate()			
		
		
	def setTensor(self, tnsr):	
	
		self.tnsr = tnsr
		
		
	#-- this function is needed only on certain IT managed machines where all traditional screen update methods fail
	def myScreenWontUpdate(self):

		#- right now only needed on IT managed MAC
		if sys.platform.lower().find('darwin') == -1:
			return
	

		if self.app != None:
			print("***** force screen update in ugly way after trying everything else *****")
			sz = self.size()
			sz.setWidth(sz.width()+self.sclast)
			self.sclast *= -1
			self.app.processEvents()
			#self.setFixedSize(sz)
			self.resize(sz)
		
		
			
		

#######################################################################
if __name__ == "__main__":

	app = QApplication(sys.argv)
		
	tensor = np.random.rand(20,400,400)
	window = imageMovieViewer()
	window.setTensor(tensor)	
			
	window.show()		
	sys.exit(app.exec_())

#######################################################################
