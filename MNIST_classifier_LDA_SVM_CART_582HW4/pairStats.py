import pickle, numpy as np
E = pickle.load(open('data/pairwiseError.dict','rb'))
M1,M2,M3 = np.zeros((10,10)), np.zeros((10,10)), np.zeros((10,10))
tbest1,tbest2,tbest3,tworst1,tworst2,tworst3 = 999,999,999,0,0,0
tb1,tb2,tb3,tw1,tw2,tw3 = None,None,None,None,None,None
for m in E:
	for k in E[m]:
		eEv, eTr = (k[1]/k[2])*100.0, (k[3]/k[4])*100.0
		print(m, k[0], eEv, eTr)
		if m == 1:  
			M1[ k[0][0],k[0][1] ] = eEv
			if eTr > tworst1: 
				tworst1 = eTr
				tw1 = k[0]
			if eTr < tbest1:
				tbest1 = eTr
				tb1 = k[0]
		elif m==2:
			M2[ k[0][0],k[0][1] ] = eEv
			if eTr > tworst2: 
				tworst2 = eTr
				tw2 = k[0]
			if eTr < tbest2:
				tbest2 = eTr
				tb2 = k[0]
		elif m==3:
			M3[ k[0][0],k[0][1] ] = eEv
			if eTr > tworst3: 
				tworst3 = eTr
				tw3 = k[0]
			if eTr < tbest3:
				tbest3 = eTr
				tb3 = k[0]	
print('LDA best train', tb1, tbest1, 'SVM best train', tb2, tbest2, 'CTree best train', tb3, tbest3)
print('LDA worst train', tw1, tworst1, 'SVM worst train', tw2, tworst2, 'CTree worst train', tw3, tworst3)
print(np.sum(M1), np.sum(M2), np.sum(M3))
print('--------  1  -------')	
for n, row in enumerate(M1):
	szTex = "%d & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f \\\\" % (n,\
		row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9])
	print(szTex)	
print('--------  2  -------')	
for n, row in enumerate(M2):
	szTex = "%d & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f \\\\" % (n,\
		row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9])
	print(szTex)	
print('--------  3  -------')	
for n, row in enumerate(M3):
	szTex = "%d & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f & %3.1f \\\\" % (n,\
		row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9])
	print(szTex)


			