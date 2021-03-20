# AP Ruymgaart, functions for audio file analysis
import math,scipy,os,sys,copy,numpy as np,scipy.io as sio,matplotlib.pyplot as plt 
from pydub import AudioSegment
from pydub.playback import play
from pydub.utils import mediainfo 
from scipy.signal import resample, butter, lfilter, freqz
from scipy.interpolate import interp1d 
from scipy.io import wavfile
from audiolazy import *
fft, fftshift, ifft = np.fft.fft, np.fft.fftshift, np.fft.ifft
abs, rndm, exp, machineEpsilon = np.absolute, np.random.normal, np.exp, np.finfo(np.float32).eps
np.set_printoptions(precision=3)

def audioread(fname):
	audio, info = AudioSegment.from_file(fname), mediainfo(fname)
	return [np.array(audio.get_array_of_samples()), float(info['sample_rate'])]
	
def np2audio(x, sr, normalized=False):	
	channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
	if normalized:  y = np.int16(x * 2 ** 15)
	else: y = np.int16(x)
	return AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)		

def writeAudiofile(data,f,fout='output.wav'): wavfile.write(fout, f, data.astype(np.int16))	

def ResampleLinear1D(original, targetLen):
	original = np.array(original, dtype=np.float)
	index_arr = np.linspace(0, len(original)-1, num=targetLen, dtype=np.float)
	index_floor = np.array(index_arr, dtype=np.int)
	index_ceil, index_rem = index_floor + 1,   index_arr - index_floor
	val1,val2 = original[index_floor], original[index_ceil % len(original)]
	interp = val1 * (1.0-index_rem) + val2 * index_rem
	assert(len(interp) == targetLen)
	return interp
	
def pad(S, plen, wh='b'):
	pd = [0.0]*plen
	if wh == 'b': return np.array( pd + list(S) + pd )
	elif wh == 'r': return np.array( list(S) + pd )
	else : return np.array( pd + list(S))

def stft(S, w=0.004, nrModes=1024, start=0,end=1000, win='shannon'):
	x, hp = np.arange(nrModes), nrModes/2 
	if win == 'shannon':
		g = np.zeros(x.shape)
		g[int(hp-w):int(hp+w)] = 1.0
	else:
		g = exp(-w*(x - hp)**2)
	img = []
	for k in range(start,end):
		try:
			win = S[k-int(hp):k+int(hp)] * g
			ftw = fftshift(fft(win))
		except:
			print('*ERR:', len(S), k-int(hp), k+int(hp), start, end)
			ftw = [0] * nrModes	
		img.append(ftw)
	return np.array(img)
	
def getNotes(img, freqs, S, cut=0.001):
	mxSig = np.max(np.abs(S))
	notes = {}
	for j,v in enumerate(img):
		indMx = np.argmax(v)       #- index of max intensity wavenumber
		mf = abs(freqs[indMx])     #- frequency of max intensity
		noteStr = freq2str(mf)     #- get note (audiolazy)
		if noteStr.find('-') > -1: #- look for e.g. C4-50% or C4+33%
			try: note = noteStr.split('-')[0]
			except: note = '?'
		elif noteStr.find('+') > -1:
			try: note = noteStr.split('+')[0]
			except: note = '?'
		try: 
			if abs(S[j])/mxSig > cut:  
				notes[note].append(j)
		except: notes[note] =[j]
	return notes
 
def getMultiNotes(img, freqs, S, cut=0.001, nn=10):
	mxSig = np.max(np.abs(S))
	notes = {}
	for j,v in enumerate(img):
		indsMx = list(np.argsort(v))   #- indexes of max intensity wavenumber
		indsMx = indsMx[::-1]          #- reverse argsort (descending)
		for iMx in indsMx[0:nn]:       #- loop over top 10 max intens wavnr inds
			mf = abs(freqs[iMx])       #- frequency of max intensity
			noteStr = freq2str(mf)     #- get note (audiolazy)
			if noteStr.find('-') > -1: #- look for e.g. C4-50% or C4+33%
				try: note = noteStr.split('-')[0]
				except: note = '?'
			elif noteStr.find('+') > -1:
				try: note = noteStr.split('+')[0]
				except: note = '?'
			try: 
				if abs(S[j])/mxSig > cut:  
					notes[note].append(j)
			except: notes[note] = [j]
	return notes		
	 
def noteDensityImg(notes, seglen, top=6):
	IMGNOTES = np.zeros((top,seglen))
	noteNames = []
	nrFound = []	
	for note in notes:
		nrFound.append( len(notes[note]) )
	indxs = list(np.argsort(nrFound))
	indxs = indxs[::-1]
	noteKeys = [*notes]
	for g,m in enumerate(indxs[0:top]):
		note = noteKeys[m]
		noteNames.append(note)
		for j in notes[note]:
			IMGNOTES[g,j] = 1
	return [IMGNOTES, noteNames]
	
def butter_pass(cutoff, fs, order=5, btype='low'):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a

def butter_filter(data, cutoff, fs, order=5, btype='low'):
    b, a = butter_pass(cutoff, fs, order=order, btype=btype)
    y = lfilter(b, a, data)
    return y

def stftImgCrop(img, freqs, low, high):
	ret,frt = [],[]
	for k in range(len(freqs)):
		if freqs[k] >= low and freqs[k] <= high:
			ret.append(img[:,k])
			frt.append(freqs[k])
	return np.flipud(np.array(ret)), frt[::-1]	

def stftPlot(img, freqs, t1, t2, aspect=20, modLbl=32, fname='stftPlot.png'):
	fig, ax = plt.subplots()
	xtickpos = np.linspace(0,img.shape[1],10).astype(int)
	xlabs = np.round(np.linspace(t1,t2, 10),2)
	for m in range(len(xlabs)): xlabs[m] = str(xlabs[m])
	ax.set_xticks(xtickpos)
	ax.set_xticklabels(xlabs)
	ylabs,yt = [],[]
	for m in range(len(freqs)): 
		if m % modLbl == 0.0:
			ylabs.append(int(round(freqs[m],0))) 
			yt.append(m)
	ax.set_yticks(yt)
	ax.set_yticklabels(ylabs)
	ax.imshow(abs(img), cmap='gist_ncar', aspect=aspect)
	plt.savefig(fname, dpi=300, bbox_inches='tight')
	plt.clf()
		
def signalInfo(S, freq):
	sLength = 1/freq          # single sample point length (time in seconds)
	duration = len(S)*sLength # duration is in seconds
	print('==== signal info ====\n\tLen signal (nr points)', len(S), 'sampling rate (freq Hz - cycles/sec)', freq)
	print('\tduration', duration,'\n\tPoint length (seconds)', sLength)
	return [duration, sLength]

def notesPlot(img, t1,t2, noteNames, fname='notesPlot.png', noteDict=None):
	fig, ax = plt.subplots()
	ax.set_xticks( np.linspace(0,img.shape[1],10).astype(int) )
	xlabs = np.round(np.linspace(t1,t2, 10),2)
	for m in range(len(xlabs)): xlabs[m] = str(xlabs[m])
	ax.set_xticklabels(xlabs)
	if not noteDict is None:
		imgN = np.zeros((len(noteDict),len(img[0])))
		for j,ntn in enumerate(noteNames):
			try: imgN[noteDict[ntn], :] = img[j,:]
			except: print('note out of given range', ntn)
		ax.imshow(np.flipud(imgN), aspect=400, cmap='Greys')
		ax.set_yticks(np.arange(len( [*noteDict] )))
		ax.set_yticklabels([*noteDict][::-1])
		for m in range(len([*noteDict])): ax.axhline(m+0.5, linewidth=0.5)		
	else:
		ax.imshow(img, aspect=400, cmap='Greys')
		ax.set_yticks(np.arange(len(noteNames)))
		ax.set_yticklabels(noteNames)
		for m in range(len(noteNames)): ax.axhline(m+0.5, linewidth=0.5)
	plt.savefig(fname, dpi=300, bbox_inches='tight')
	plt.clf()

def noteImg2noteList(IMGNOTES, pLength, noteNames, offset=0, nSeg=15):
	musicScore = []
	intv = len(IMGNOTES[0,:])/nSeg
	for si in range(nSeg):
		seg = IMGNOTES[:, int(si*intv):int((si+1)*intv)]
		segTime = offset + ((int(si*intv) + int((si+1)*intv))/2) * pLength
		noteDensity = []
		for nn in range(len(noteNames)): noteDensity.append(np.sum(seg[nn,:]))
		musicScore.append([segTime, noteNames[np.argmax(np.array(noteDensity))] ])
	return 	musicScore
	
def compressScore(score):
	tLast,nLast = score[0][0], score[0][1]
	compressed = [[tLast,nLast]]	
	for nt in score:
		t,n = nt[0],nt[1]
		if n != nLast: compressed.append(nt)
		nLast = n
	return compressed
	
def processCmdArgs(arglst):
	cmd = {}
	for c in arglst:
		try:
			elms = c.split('=')
			cmd[elms[0]] = elms[1]
		except: pass
	return cmd

def makeNoteDictionary(strt='C4', stp='G5'):
	n, keep = 0, False
	dNotes = {}
	for octv in [1,2,3,4,5]:
		for Nt in ['C','D','E','F','G','A','B']:
			for shrp in ['','#']:
				note = Nt+shrp+str(octv)
				if note == strt: keep = True
				if keep: 
					dNotes[note] = n
					n += 1
				if note == stp: keep = False	
	return dNotes