from sympy import *
import numpy as np

B,th1,th2,ph1,ph2,yt,et = symbols('B theta_1 theta_2 phi_1 phi_2 y_t epsilon_t')

# https://online.stat.psu.edu/stat510/lesson/4/4.1 


# https://stats.stackexchange.com/questions/374460/how-do-i-write-a-mathematical-equation-for-seasonal-arima-0-0-1-x-2-1-2-peri

'''
Example:
ARIMA(0,0,1)x(2,1,2)
'''

SAR2_12 = (1 - ph1*B**12 - ph2*B**24)  
SMA2_12 = (1 + th1*B**12 + th2*B**24)
SI      = (1 - B**12)                  # seasonal integration "i" term
MA1     = (1 + th1*B)

mdlLHS = SAR2_12 * SI * yt
mdlRHS = MA1 * SMA2_12 * et
print(latex(expand(mdlLHS)))
print('=')
print(latex(expand(mdlRHS)))