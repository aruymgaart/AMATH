rm images/*.png images/score.txt

#=== Floyd Bass ====
if false
then
python3 musicScore.py \
file=Floyd.m4a \
modes=1024 win='shannon' ww=40 \
lowpass=True,150.0 highpass=False,50.0 downsample=True,5 \
seglen=14000 maxNotesSeg=5 resolutionSeg=25 icut=0.01 multinote=1 \
plots=True preview=False plotscale=False,'C1','C6' \
stftPlot=40,170,30,32 saveAndExit=False,floydbass.wav
fi

#=== Floyd guitar ====
if false
then
python3 musicScore.py \
file=Floyd.m4a \
modes=1024 win='shannon' ww=80 \
lowpass=True,900.0 highpass=True,300.0 downsample=True,15 \
seglen=15000 maxNotesSeg=14 resolutionSeg=50 icut=0.04 multinote=4 \
plots=True preview=False plotscale=True,'C4','G5' \
stftPlot=300,900,20,32 saveAndExit=False,floydguitar.wav
fi

#=== GNR guitar ===
if true
then
python3 musicScore.py \
file=GNR.m4a \
modes=1024 win='shannon' ww=40 \
lowpass=False,150.0 highpass=True,250.0 downsample=True,13 \
seglen=8000 maxNotesSeg=6 resolutionSeg=25 icut=0.01 multinote=1 \
plots=True preview=False plotscale=False,'C1','C6' \
stftPlot=180,780,10,32 saveAndExit=False,gnrguitar.wav
fi