# AP Ruymgaart 
# Example hierarchical forecasting
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
mx = np.matmul
'''
see https://otexts.com/fpp2/reconciliation.html#reconciliation
1) produce forecasts
2) obtain forecast residual vectors ve ()
3) build W = sum_{residVecs} ve*ve'
4) build G = (S' W^-1 S)^-1 S' W^-1
5) Reconciled forecasts: y = SGy_h = S((S' W^-1 S)^-1 S' W^-1)y_h

in  this example  (like in MinT slides)
3 base forecasts, 1 summed frecast
    S = (4x3) 
    Sb = (4x3)x(3x1) = (4x1)
    y,yh = (4x1)
    G = (3x4)
    W = (4x4)
    (S' W^-1 S)^-1 S' W^-1 = ( (3x4)x(4x4)
    Pr = SG = (4x3)x(3x4) = (4x4)
'''

def getSales(df, store, dept):
    df1 = df[df['Store'] == store]
    df1 = df1[df1['Dept'] == dept]
    return df1['Weekly_Sales'].to_numpy()

def reconciliationMatrix(S,W):
    iW = np.linalg.inv(W)
    StWiS = mx(S.T,  mx(iW,S))
    StWiS_inv = np.linalg.inv(StWiS)
    StWi = mx(S.T, iW)
    return mx(StWiS_inv, StWi) 


if __name__ == '__main__':
    if False:
        S = np.array([ [1,1],[1,0],[0,1] ])
        W = np.eye(3)
        G = reconciliationMatrix(S,W)
        print('reconciliationMatrix', G)
        print('projector', mx(S,G))
        exit()


    df = pd.read_csv('walmart_sample_data/train.csv')
    curve_1 = getSales(df, 1, 1)
    curve_2 = getSales(df, 1, 2)
    curve_3 = getSales(df, 1, 3)
    curve_ttl = curve_1 + curve_2 + curve_3
    Nd = len(curve_1)
    nPred = 26

    #---- sum matrix S ----
    # there are 3 base forecasts
    S = np.array([[1,1,1],[1,0,0],[0,1,0],[0,0,1]])

    #---- forecast fits ----
    mod_1 = SARIMAX(curve_1[0:Nd-nPred], order=(0,0,0), seasonal_order=(1,1,1,52))
    res_1 = mod_1.fit()
    residual_1, fit_1 = res_1.resid, res_1.fittedvalues

    mod_2 = SARIMAX(curve_2[0:Nd-nPred], order=(0,1,0), seasonal_order=(1,1,1,52))
    res_2 = mod_2.fit()
    residual_2, fit_2 = res_2.resid, res_2.fittedvalues

    mod_3 = SARIMAX(curve_3[0:Nd-nPred], order=(0,1,0), seasonal_order=(1,1,1,52))
    res_3 = mod_3.fit()
    residual_3, fit_3 = res_3.resid, res_3.fittedvalues

    mod_sum = SARIMAX(curve_ttl[0:Nd-nPred], order=(0,0,0), seasonal_order=(1,1,1,52))
    res_sum = mod_sum.fit()
    residual_sum, fit_sum = res_sum.resid, res_sum.fittedvalues

    #---- forecast future points ----
    pred_1 = res_1.forecast(nPred) 
    model_1 = np.array(list(fit_1) + list(pred_1))

    pred_2 = res_2.forecast(nPred) 
    model_2 = np.array(list(fit_2) + list(pred_2))

    pred_3 = res_3.forecast(nPred) 
    model_3 = np.array(list(fit_3) + list(pred_3))

    pred_sum = res_sum.forecast(nPred) 
    model_sum = np.array(list(fit_sum) + list(pred_sum))

    sum_base_models = model_1 + model_2 + model_3

    yh = np.array( [model_sum, model_1, model_2, model_3] )


    if True:
        A = np.array([residual_1[55:len(residual_1)],residual_2[55:len(residual_1)],residual_3[55:len(residual_1)],residual_sum[55:len(residual_1)]])
    else:
        res_1 = pred_1 - curve_1[Nd-nPred-1:-1]
        res_2 = pred_2 - curve_2[Nd-nPred-1:-1]
        res_3 = pred_3 - curve_3[Nd-nPred-1:-1]
        res_4 = pred_sum - curve_ttl[Nd-nPred-1:-1]
        A = np.array([res_1,res_2,res_3,res_4])

    if False:
        Cov = np.cov(A)
        Cv = np.zeros(Cov.shape)
        for k in range(4): Cv[k,k] = Cov[k,k]*1.0
    else:
        Cv = np.eye(4)*2

    if False: # G = (3x4)
        G = Gp
    elif False:
        G = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    else:
        G = reconciliationMatrix(S,Cv)

    #================= OUTPUT ==================
    print(res_1.summary())
    print(res_2.summary())
    print(res_3.summary())
    print(res_sum.summary())
    #- projector
    P = mx(S,G)
    print('COV', Cv)
    #print('StWiS', StWiS)
    print('G', G)
    print('P', P)
    #- reconciliated
    yR = mx(P, yh)

    if False:
        plt.plot(residual_1, label='res 1')
        plt.plot(residual_2, label='res 2')
        plt.plot(residual_3, label='res 3')
        plt.plot(residual_sum, label='res sum')
        plt.legend()
        plt.show()


    #plt.plot(p, label='ARIMA, order=(%d,%d,%d) predicted' % (2, 1, 0))
    if False:
        plt.plot(curve_1, label='True 1')
        plt.plot(model_1, label='fit/fcst 1')
        plt.plot(curve_2, label='True 2')
        plt.plot(model_2, label='fit/fcst 2')
        plt.plot(curve_3, label='True 3')
        plt.plot(model_3, label='fit/fcst 3')
    else:
        pass

    fig, ax = plt.subplots(4,1)
    T = np.arange(len(curve_ttl))
    ax[0].plot(T[60:-1], curve_ttl[60:-1], label='True sum')
    ax[0].plot(T[60:-1], sum_base_models[60:-1], label='sum of indiv base models')
    ax[0].plot(T[60:-1], yh[0][60:-1], label='fit/fcst sum')
    ax[0].plot(T[60:-1], yR[0][60:-1], label='reconciliated')
    ax[0].axvline(x=len(curve_ttl)-nPred, c='black')

    ax[1].plot(T[60:-1], curve_1[60:-1], label='curve 1 true')
    ax[1].plot(T[60:-1], yR[1][60:-1], label='curve 1 reconciliated')
    ax[1].plot(T[60:-1], yh[1][60:-1], label='curve 1 model')
    ax[1].axvline(x=len(curve_ttl)-nPred, c='black')

    ax[2].plot(T[60:-1], curve_2[60:-1], label='curve 2 true')
    ax[2].plot(T[60:-1], yR[2][60:-1], label='curve 2 reconciliated')
    ax[2].plot(T[60:-1], yh[2][60:-1], label='curve 2 model')
    ax[2].axvline(x=len(curve_ttl)-nPred, c='black')

    ax[3].plot(T[60:-1], curve_3[60:-1], label='curve 3 true')
    ax[3].plot(T[60:-1], yR[3][60:-1], label='curve 3 reconciliated')
    ax[3].plot(T[60:-1], yh[3][60:-1], label='curve 3 model')
    ax[3].axvline(x=len(curve_ttl)-nPred, c='black')
    
    ax[0].legend(), ax[1].legend(), ax[2].legend(), ax[3].legend()
    plt.show()


