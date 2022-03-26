# AP Ruymgaart 
# Example hierarchical forecasting (see Hyndman et.al.)
# NOTE: not yet working as expected
import pandas as pd, numpy as np, sklearn
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import sklearn.metrics
from sklearn.metrics import mean_absolute_percentage_error
mx, mape = np.matmul, mean_absolute_percentage_error
'''
see https://otexts.com/fpp2/reconciliation.html#reconciliation
1) produce forecasts
2) obtain forecast residual vectors ve ()
3) build W = sum_{residVecs} ve*ve'
4) build G = (S' W^-1 S)^-1 S' W^-1
5) Reconciled forecasts: y = SGy_h = S((S' W^-1 S)^-1 S' W^-1)y_h

in  this example  (like in MinT slides)
3 base forecasts, 1 summed frecast
    n = nr base forecasts (3)
    m = nr total forecasts (4)
    k = m-n (1)
    S = (mxn) = (4x3) 
    Sb = (4x3)x(3x1) = (4x1)
    y,yh = (4x1)
    G = (3x4)
    W = (4x4)
    (S' W^-1 S)^-1 S' W^-1 = ( (3x4)x(4x4)
    Pr = SG = (4x3)x(3x4) = (4x4)
    S' = [C' I_n]
    U' = [I_k -C] = (k)x(m)
    J = [0_nxk I_n] = (n)x(m)
    here C = [1 1 1] = (k)x(n)
'''

def getSales(df, store, dept):
    df1 = df[df['Store'] == store]
    df1 = df1[df1['Dept'] == dept]
    return df1['Weekly_Sales'].to_numpy()

'''
reconciliation matrix
    (S' W^-1 S)^-1 S' W^-1
or equivalently (eqn. 10)
    J - JWU(U'WU)^-1U'
and the latter with pseudoinverse:
    J - JSVS'U(U'SVS'U)^-1U'
'''
def reconciliationMatrix(S,W, pseudoinverse=True):
    n = S.shape[1]
    m = S.shape[0]
    k = m - n
    if pseudoinverse:
        #print('S',S)
        C = S[0:k,:] 
        #print('C', C, C.shape)
        J = np.zeros((n,m))
        for i in range(n): J[i,i+k] = 1
        U = np.zeros((m,k))
        for i in range(k): U[i,i] = 1
        U[k:m,:] = -1.0*C.T
        WU = mx(W,U)
        UtWU = mx(U.T, WU)
        UtWU_i = np.linalg.inv(UtWU)
        R0 = mx(UtWU_i, U.T)
        R1 = mx(WU, R0)
        #print('J',J,'WU',WU, UtWU_i)
        return J - mx(J, R1)
    else:
        iW = np.linalg.inv(W)
        StWiS = mx(S.T,  mx(iW,S))
        StWiS_inv = np.linalg.inv(StWiS)
        StWi = mx(S.T, iW)
        return mx(StWiS_inv, StWi) 

def tail(s, nTail):
    return s[len(s)-nTail:len(s)]

def head(s, nTail):
    return s[0:len(s)-nTail]



if __name__ == '__main__':
    if False:
        S = np.array([ [1,1],[1,0],[0,1] ])
        W = np.eye(3)
        G = reconciliationMatrix(S,W)
        print('reconciliationMatrix:\n', G)
        print('projector:\n', mx(S,G))
        print('---- not using pseudoinverse ----')
        G = reconciliationMatrix(S,W, pseudoinverse=False)
        print('reconciliationMatrix:\n', G)
        print('projector:\n', mx(S,G))
        exit()

    df = pd.read_csv('walmart_sample_data/train.csv')
    curve_1 = getSales(df, 1, 1)
    curve_2 = getSales(df, 1, 2)
    curve_3 = getSales(df, 1, 3)
    curve_ttl = curve_1 + curve_2 + curve_3
    Nd = len(curve_1)
    nPred = 13
    W_option = 3
    cheat = True

    #---- sum matrix S ----
    # there are 3 base forecasts
    S = np.array([[1,1,1],[1,0,0],[0,1,0],[0,0,1]])
    #print(S)
    #print(S.T)
    #exit()

    #---- forecast fits ----
    mod_1 = SARIMAX(head(curve_1,nPred), order=(0,0,0), seasonal_order=(1,1,1,52))
    res_1 = mod_1.fit()
    residual_1, fit_1 = res_1.resid, res_1.fittedvalues

    mod_2 = SARIMAX(head(curve_2,nPred), order=(0,1,0), seasonal_order=(1,1,1,52))
    res_2 = mod_2.fit()
    residual_2, fit_2 = res_2.resid, res_2.fittedvalues

    mod_3 = SARIMAX(head(curve_3,nPred), order=(0,1,0), seasonal_order=(1,1,1,52))
    res_3 = mod_3.fit()
    residual_3, fit_3 = res_3.resid, res_3.fittedvalues

    mod_sum = SARIMAX(head(curve_ttl,nPred), order=(0,0,0), seasonal_order=(1,1,1,52))
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


    if cheat: #- "cheat with know true curves"
        r1 = np.abs( pred_1 - tail(curve_1, nPred) )
        r2 = np.abs( pred_2 - tail(curve_2, nPred) )
        r3 = np.abs( pred_3 - tail(curve_3, nPred) )
        r4 = np.abs( pred_sum - tail(curve_ttl, nPred) )

        sae1, sae2, sae3, sae4 = np.sum(r1), np.sum(r2), np.sum(r3), np.sum(r4)

        print('SUM ABS ERR (SAE) c1', sae1)
        print('SUM ABS ERR (SAE) c2', sae2)
        print('SUM ABS ERR (SAE) c3', sae3)
        print('SUM ABS ERR (SAE) c4', sae4)

        A = np.array([r4,r1,r2,r3])

        if True:
            plt.title('Difference from true')
            plt.plot(r1, label='1 SAE=%f' % (sae1))
            plt.plot(r2, label='2 SAE=%f' % (sae2))
            plt.plot(r3, label='3 SAE=%f' % (sae3))
            plt.plot(r4, label='4 (sum) SAE=%f' % (sae4))
            plt.legend()
            plt.show()

    else: 
        A = np.array([residual_sum[55:-1],residual_1[55:-1],residual_2[55:-1],residual_3[55:-1]])

    if W_option == 1:
        Cv = np.eye(4)*1.0
    elif W_option == 2:
        Cov = np.cov(A)
        Cv = np.zeros(Cov.shape)
        for k in range(4): Cv[k,k] = Cov[k,k]*1.0
    else:
         Cv = np.cov(A)
    
   
    G = reconciliationMatrix(S, Cv)
    P = mx(S, G)    #- projector
    yR = mx(P, yh) #- reconciliated forecasts

    #================= OUTPUT ==================
    print(res_1.summary())
    print(res_2.summary())
    print(res_3.summary())
    print(res_sum.summary())
    
    print('===== COV ===== \n', Cv)
    print('Cov matrix condition number (large means near singular and pseudoinverse should be used)', np.linalg.cond(Cv))
    print('===== G ===== \n', G)
    print('===== projector  P=SG ===== \n', P)
    

    if True:
        plt.plot(residual_1, label='res 1')
        plt.plot(residual_2, label='res 2')
        plt.plot(residual_3, label='res 3')
        plt.plot(residual_sum, label='res sum')
        plt.legend()
        plt.show()

    true_sum = curve_ttl[60:-1]
    Lcrop = len(true_sum)
    sum_models = sum_base_models[60:-1]
    model_sum = yh[0][60:-1]
    recon_sum = yR[0][60:-1]
    c1 = curve_1[60:-1]
    c2 = curve_2[60:-1]
    c3 = curve_3[60:-1]
    yh1 = yh[1][60:-1]
    yh2 = yh[2][60:-1]
    yh3 = yh[3][60:-1]
    yr1 = yR[1][60:-1]
    yr2 = yR[2][60:-1]
    yr3 = yR[3][60:-1]

    print('===== top of hierarchy level ====')
    print('MAPE sum_models', mape( tail(true_sum, nPred), tail(sum_models, nPred) ) )
    print('MAPE model_sum',  mape( tail(true_sum, nPred), tail(model_sum, nPred) ) )
    print('MAPE recon_sum',  mape( tail(true_sum, nPred), tail(recon_sum, nPred) ) )
    print('===== base levels ====')
    print('MAPE base curve 1 model',               mape( tail(c1, nPred), tail(yh1, nPred) ) )
    print('MAPE base curve 1 model reconciliated', mape( tail(c1, nPred), tail(yr1, nPred) ) )
    print('MAPE base curve 2 model',               mape( tail(c2, nPred), tail(yh2, nPred) ) )
    print('MAPE base curve 2 model reconciliated', mape( tail(c2, nPred), tail(yr2, nPred) ) )
    print('MAPE base curve 3 model',               mape( tail(c3, nPred), tail(yh3, nPred) ) )
    print('MAPE base curve 3 model reconciliated', mape( tail(c3, nPred), tail(yr3, nPred) ) )

    fig, ax = plt.subplots(4,1)
    T = np.arange(len(curve_ttl))
    ax[0].plot(T[60:-1], true_sum, label='True sum')
    ax[0].plot(T[60:-1], sum_models, label='sum of models')
    ax[0].plot(T[60:-1], model_sum, label='model of sum')
    ax[0].plot(T[60:-1], recon_sum, label='reconciliated')
    ax[0].axvline(x=len(curve_ttl)-nPred, c='black')

    ax[1].plot(T[60:-1], c1, label='curve 1 true')
    ax[1].plot(T[60:-1], yr1, label='curve 1 reconciliated')
    ax[1].plot(T[60:-1], yh1, label='curve 1 model')
    ax[1].axvline(x=len(curve_ttl)-nPred, c='black')

    ax[2].plot(T[60:-1], c2, label='curve 2 true')
    ax[2].plot(T[60:-1], yr2, label='curve 2 reconciliated')
    ax[2].plot(T[60:-1], yh2, label='curve 2 model')
    ax[2].axvline(x=len(curve_ttl)-nPred, c='black')

    ax[3].plot(T[60:-1], c3, label='curve 3 true')
    ax[3].plot(T[60:-1], yr3, label='curve 3 reconciliated')
    ax[3].plot(T[60:-1], yh3, label='curve 3 model')
    ax[3].axvline(x=len(curve_ttl)-nPred, c='black')

    ax[0].legend(), ax[1].legend(), ax[2].legend(), ax[3].legend()
    plt.show()


