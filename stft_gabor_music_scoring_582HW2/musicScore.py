# AP Ruymgaart, main music scoring script
import math,scipy,os,sys,copy,numpy as np,scipy.io as sio,matplotlib.pyplot as plt, prettytable
from scipy.signal import butter, lfilter, freqz
from sklearn.cluster import MeanShift, estimate_bandwidth 
from pydub import AudioSegment
from pydub.playback import play
from pydub.utils import mediainfo 
from scipy.signal import resample
from scipy.interpolate import interp1d
from audiolazy import *
fft, fftshift, ifft = np.fft.fft, np.fft.fftshift, np.fft.ifft
abs, rndm, exp = np.absolute, np.random.normal, np.exp
machineEpsilon = np.finfo(np.float32).eps
np.set_printoptions(precision=3)
from audioFunctions import *

#====== SETUP ======
print('===================================== start =================================\nInput:')
cmds = processCmdArgs(sys.argv)
for c in cmds: print('\t', c, cmds[c])
try:	
	n,pi = int(cmds['modes']), np.pi
	w, wintype = float(cmds['ww']), cmds['win']
	cutoffLow, cutoffHigh = float(cmds['lowpass'].split(',')[1]), float(cmds['highpass'].split(',')[1]) 
	lowPass, highPass  = cmds['lowpass'].split(',')[0]=='True', cmds['highpass'].split(',')[0]=='True'
	makePlots = cmds['plots'] == 'True'
	downSample, dsmplFact = cmds['downsample'].split(',')[0]=='True', float(cmds['downsample'].split(',')[1])
	piecelengthWanted = int(cmds['seglen'])
	plScale,ntFirst,ntLast = cmds['plotscale'].split(',')[0]=='True', cmds['plotscale'].split(',')[1], \
		cmds['plotscale'].split(',')[2]
	mxNotesPerSeg = int(cmds['maxNotesSeg'])
	icut = float(cmds['icut'])
	imultinote = int(cmds['multinote'])
	resolutionPerSeg = int(cmds['resolutionSeg'])
	saveAndExit, fout = cmds['saveAndExit'].split(',')[0] == 'True', cmds['saveAndExit'].split(',')[1]
	preview = cmds['preview'] == 'True'    # preview = listen and exit
	[S,freq] = audioread(cmds['file'])
	stftPlotPars = [ int(cmds['stftPlot'].split(',')[0]), int(cmds['stftPlot'].split(',')[1]), \
		int(cmds['stftPlot'].split(',')[2]), int(cmds['stftPlot'].split(',')[3]) ]
	if plScale: nd = makeNoteDictionary(strt=ntFirst, stp=ntLast)
	else: nd = None	
except:
	print('INPUT ERROR', cmds)
	exit()	

#====== downsample & segment =====
if downSample:
	S = ResampleLinear1D(S, int(len(S)/dsmplFact)) 
	freq = freq/dsmplFact
nrPieces = int(len(S)/piecelengthWanted)
piecelength = int(len(S)/nrPieces)
reclen = piecelength*nrPieces
print(nrPieces, piecelength, reclen, len(S))
if len(S) > reclen: S = S[0:reclen]
if len(S) < reclen: S = pad(S, reclen-len(S), wh='r')
	
#====== FFT scaling =====
[duration, pLength] = signalInfo(S,freq)
L = (n * pLength)/2
x2 = np.linspace(-L,L,n+1)
x = x2[0:n]
scale = (2*pi)/(2*L)
nk = np.append(np.arange(0,n/2),np.arange(-n/2,0)) 
k = scale * nk
ks = fftshift(k)
nks = fftshift(nk)
freqs = (1/(2*pi)) * ks

#====== FILTER (if both lowPass, highPass then bandpass) =====
order = 6
if lowPass:  S = butter_filter(S, cutoffLow, freq, order, btype='low')
if highPass: S = butter_filter(S, cutoffHigh, freq, order, btype='high')

#====== info output & if preview then listen and quit ======	
if downSample: print('DOWNSAMPLED to', freq, 'factor=', dsmplFact, 'original sample freq=', freq*dsmplFact)
if lowPass: print('LOW PASS FILTER', cutoffLow)
if highPass: print('HIGH PASS FILTER', cutoffHigh)
print('Clip segmented into', nrPieces, 'pieces of length', piecelength, reclen, len(S))	
print('Interval width=', 2*L, 'L=',L, 'lowest freq =', 1/L, 'highest freq', 0.5/pLength)
if True:
	tbl = prettytable.PrettyTable()
	tbl.field_names = ["Wavenumber", "index", "wavelen in time (2L/n)", "freq"]
	print('==== FT info ===\n', 'Nr Fourier modes=', n)
	for j in range(len(ks)):
		if j % 64 == 0.0 and nks[j]: tbl.add_row([ ks[j], nks[j], (2*L)/nks[j],  freqs[j] ])
	print(tbl)
if preview:
	play(np2audio(S, freq))
	exit()
if saveAndExit:
	writeAudiofile(S, int(freq), fout)
	exit()

#====== MAIN PROCESSING LOOP =========
pS = pad(S, int(n/2))
wholeScore = []
for ns in range(nrPieces):
	strt,stp = piecelength*ns, piecelength*(ns+1)
	img = stft(pS, w=w, nrModes=n, start=strt+int(n/2),end=stp+int(n/2), win=wintype)
	notes = getMultiNotes(img, freqs, S, icut, imultinote)
	[densImg, noteNames] = noteDensityImg(notes, piecelength, top=mxNotesPerSeg)
	musicScore = noteImg2noteList(densImg, pLength, noteNames, offset=strt*pLength, nSeg=resolutionPerSeg)
	wholeScore += musicScore
	if makePlots: 
		imgCrop, freqCrop = stftImgCrop(img, freqs, stftPlotPars[0], stftPlotPars[1])
		stftPlot(imgCrop, freqCrop, strt*pLength, stp*pLength, aspect=stftPlotPars[2], modLbl=stftPlotPars[3], \
			fname='images/stft_seg%d.png' % (ns+1))
		notesPlot(densImg, strt*pLength, stp*pLength, noteNames, fname='images/noteDensity_seg%d.png' % (ns+1), noteDict=nd)

fout = open('images/score.txt','w') 
compr = compressScore(wholeScore)
print('=== compressed ===')
for j,nt in enumerate(compr): 
	print(j,nt)
	fout.write( "%5.2f, %s\n" %(nt[0],nt[1]) )
fout.close()	



