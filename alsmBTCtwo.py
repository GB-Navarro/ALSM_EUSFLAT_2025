# ALSM two-lag EUSFLAT 2025

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import time
from numpy import linalg as la
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
import seaborn as sns

#sns.set()  # style='darkgrid'

# load data

df = pd.read_excel('dataBTC.xlsx')

data = df['y'].to_numpy()   


delay = 2   # delay (lags)
n = 999     # linha do arquivo data onde está o último dado de treinamento (data 999)
m = 2524    # linha do arquivo data onde o último dado de teste (data 2524)

Xtrain = np.zeros((n-delay+1,delay))
ytrain = np.zeros(n-delay+1)
Xtest  = np.zeros((m-n,delay))
ytest  = np.zeros(m-n)

# Assemble training and test data

for i in range (0, n-delay+1):
    for j in range(0,delay):
        Xtrain[i,j] = data[i+j]
    ytrain[i] = data[i+delay]

for i in range(0,m-n):
    for j in range(0,delay):
        Xtest[i,j] = data[n-delay+1+i+j]
    ytest[i] = data[n+i+1]


plt.figure()            # to visualize data
plt.plot(ytrain)
plt.xlabel('k')
plt.ylabel('y')
plt.title('Train data')
#plt.ylim([0, 2])
#plt.show()

plt.figure()
plt.plot(ytest)
plt.xlabel('k')
plt.ylabel('y')
plt.title('Test data')
#plt.ylim([0, 2])
#plt.show()

def gaussmf(x, c, s):
    aux = (x - c)/s
    z = np.exp(-(aux*aux)/2)
    return z

def auto_gauss_granules(rules_number, x):
    pass

def gaussgranules(x):
    
    x1 = x[0]
    x2 = x[1]

    sx1 = 0.55
    sx2 = 0.40

    c11 =  0.10; s11 = sx1
#    c12 =  0.40; s12 = sx1
    c13 =  0.80; s13 = sx1
#    c14 =  0.80; s14 = sx1

    c21 =  0.20; s21 = sx2
#    c22 =  0.40; s22 = sx2
    c23 =  0.60; s23 = sx2
#    c24 =  0.80; s24 = sx2

    
    r11 = gaussmf(x1,c11,sx1)*gaussmf(x2,c21,sx2)

#    r22 = gaussmf(x1,c12,sx1)*gaussmf(x2,c22,sx2)

    r33 = gaussmf(x1,c13,s11)*gaussmf(x2,c23,sx2)

#    r44 = gaussmf(x1,c14,sx1)*gaussmf(x2,c24,sx2)

#    r55 = gaussmf(x1,c12,sx1)*gaussmf(x2,c24,sx2)
    
#    r66 = gaussmf(x1,c12,s12)*gaussmf(x2,c21,s21)
    
#    r = np.array([r11, r12, r13, r14, r15, \
#                  r21, r22, r23, r24, r25, \
#                  r31, r32, r33, r34, r35, \
#                  r41, r42, r43, r44, r45, \
#                  r51, r52, r53, r54, r55])

    d = np.array([r11, r33])
    
    return d


r = gaussgranules(Xtrain[0]).size

d = np.zeros(r)                 # membership degrees of inputs

ndatatrain = Xtrain.shape[0]
ndatatest  = Xtest.shape[0]
ndata = ndatatrain + ndatatest


o = np.zeros((ndata,))          # model outputs
OT = np.zeros((ndatatrain,))    # outputs training data
OS = np.zeros((ndatatest,))     # outputs testing  data

lamb = 0.99  # 0.99             # forgetting factor
alfa = 1000
DA = np.zeros((ndata,2*r))

P = alfa*np.eye(2*r)
G = alfa*np.eye(2*r)
a = np.zeros((2*r,))
u = np.zeros((2*r,))

norm = np.zeros((ndatatrain,))

kc = 1/np.sqrt(2*np.pi)             # krnel constant
ze = 1.0                            # zeta in the range [0.1, 2] Rong 2019
kz = kc*(1/(ze*ze*ze))
zs = 1/(2*ze*ze)

# start measuring cpu training time

st = time.process_time()

# training 

for t in range(ndatatrain):
    x = Xtrain[t]
    d = gaussgranules(x)
    sumd = sum(d)
    for j in range(r):
        a[2*j] = d[j]*d[j]/sumd
        a[2*j+1] = d[j]/sumd
    sq = (np.absolute(ytrain[t] - np.dot(a,u)))*(np.absolute(ytrain[t] - np.dot(a,u)))

#    psi = 1

    psi =  kz*np.exp(-(sq/zs))
    b = lamb + psi*np.dot(np.dot(a, P), a)
    P = (P - (psi*np.outer(np.dot(P,a), np.dot(a,P)))/b)/lamb
    u = u + psi*np.dot(P,a)*(ytrain[t] - np.dot(a,u))    # model parameters
    o = np.dot(a,u)                                      # model output 
    OT[t] = o
    norm[t] = la.norm(u)

et = time.process_time()

# get execution time training

res = et - st

print('cpu execution time training:', res, 'seconds')

plt.figure()
plt.plot(norm, label="norm")
plt.xlabel('steps')
plt.ylabel('Norm')
plt.title('Norm')           # norm of the vector u (just to watch)
plt.show()

# testing 

for t in range(ndatatest):
    x = Xtest[t]
    d = gaussgranules(x)
    sumd = sum(d)
    for j in range(r):
        a[2*j] = d[j]*d[j]/sumd
        a[2*j+1] = d[j]/sumd

    b = lamb + np.dot(np.dot(a, P), a)
    P = (P - (np.outer(np.dot(P,a), np.dot(a,P)))/b)/lamb    
    u = u + np.dot(P,a)*(ytest[t] - np.dot(a,u))    # model parameters recursive
    
    o = np.dot(a,u)                                 # model output recursive 
    OS[t] = o


print('mse  train =', mean_squared_error(ytrain, OT))
print('nrms train =', root_mean_squared_error(ytrain, OT)/ytrain.std())
print('rmse train =', root_mean_squared_error(ytrain, OT))
print('mae  train =', mean_absolute_error(ytrain, OT))

plt.figure()
plt.plot(ytrain, label="Actual")
plt.plot(OT, label="ALSM")
plt.legend(loc='upper right')
plt.xlabel('k')
plt.ylabel('y')
plt.title('Forecast (train)')
#plt.ylim([1, 3])


print('MSE   test =', mean_squared_error(ytest, OS))
print('NRMSE test =', root_mean_squared_error(ytest, OS)/ytest.std())
print('RMSE  test =', root_mean_squared_error(ytest, OS))
print('MAE   test =', mean_absolute_error(ytest, OS))

plt.figure()
plt.plot(ytest, label="Actual")
plt.plot(OS, label="ALSM2")
plt.legend(loc='upper right')
plt.xlabel('day')
plt.ylabel('RV')
#plt.title('Forecast (test)')
#plt.ylim([0, 2])

plt.show()



