# ALSM 3 inputs EUSFLAT 2025

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

# sns.set()  # style='darkgrid'

# load data

dfXtrain = pd.read_excel('XtrainBTC.xlsx')
dfXtest = pd.read_excel('XtestBTC.xlsx')
dfYtrain = pd.read_excel('ytrainBTC.xlsx')
dfYtest = pd.read_excel('ytestBTC.xlsx')

Xtrain = dfXtrain.to_numpy()
Xtest = dfXtest.to_numpy()
ytrain = dfYtrain.to_numpy()
ytest = dfYtest.to_numpy()

ndatatrain = len(Xtrain)
ndatatest = len(Xtest)

inputs = 3

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

def gaussgranules(x, c, s, nc, ni):
    
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]

    c11 = c[0][0]; s11 = s[0][0]      
    c12 = c[1][0]; s12 = s[1][0]      
#    c13 = c[2][0]; s13 = s[2][0]
#    c14 =  0.00; s14 = sx1

    c21 =  c[0][1]; s21 = s[0][1]      
    c22 =  c[1][1]; s22 = s[1][1]      
#    c23 =  c[2][1]; s23 = s[2][1]
#    c24 =  2.0; s24 = sx2

    c31 =  c[0][2]; s31 = s[0][2]      
    c32 =  c[1][2]; s32 = s[1][2]      
#    c33 =  0.30; s33 = sx3
#    c34 =  2.0; s24 = sx2
    
    r11 = gaussmf(x1,c11,s11)*gaussmf(x2,c21,s21)*gaussmf(x3,c31,s31) 

    r22 = gaussmf(x1,c12,s12)*gaussmf(x2,c22,s22)*gaussmf(x3,c32,s32) 
    

#    r33 = gaussmf(x1,c13,s13)*gaussmf(x2,c23,s23)*gaussmf(x3,c33,s33) 

#    r44 = gaussmf(x1,c12,s12)*gaussmf(x2,c22,s22)*gaussmf(x3,c32,s32)

#    r55 = gaussmf(x1,c11,s12)*gaussmf(x2,c22,s22)*gaussmf(x3,c31,s31)
    
#    r66 = gaussmf(x1,c12,s12)*gaussmf(x2,c22,s21)*gaussmf(x3,c31,s32)
    
#    r = np.array([r11, r12, r13, r14, r15, \
#                  r21, r22, r23, r24, r25, \
#                  r31, r32, r33, r34, r35, \
#                  r41, r42, r43, r44, r45, \
#                  r51, r52, r53, r54, r55])

    d = np.array([r11, r22])
    
    return d

ni = Xtrain.shape[1]        # number of inputs
nc = 4                      # number of centers and spreads of Gaussians

c = np.zeros((nc, ni))
s = np.zeros((nc, ni))

# Initialization of centers and spread if gradient is used

sx1 = 0.20
sx2 = 0.60
sx3 = 0.05

c[0][0] = 0.10;  s[0][0] = sx1    # c11     
c[1][0] = 0.85;  s[1][0] = sx2    # c12                                      

c[0][1] = 0.10;  s[0][1] = sx1    # c21                
c[1][1] = 0.70;  s[1][1] = sx2    # c22

c[0][2] = 0.10;  s[0][2] = sx3    # c31                
c[1][2] = 0.20;  s[1][2] = sx3    # c32

r = gaussgranules(Xtrain[0], c, s, nc, ni).size

d = np.zeros(r)                   # membership degrees of inputs 

ndata = ndatatrain + ndatatest

o = np.zeros((ndata,))          # model outputs 
OT = np.zeros((ndatatrain,))    # outputs training data
OS = np.zeros((ndatatest,))     # outputs testing data

lamb = 0.99
alfa = 1000
DA = np.zeros((ndata,2*r))

P = alfa*np.eye(2*r)
G = alfa*np.eye(2*r)
a = np.zeros((2*r,))
u = np.zeros((2*r,))

norm = np.zeros((ndatatrain,))

kc = 1/np.sqrt(2*np.pi)             # krnel constant
ze = 0.41                           # zeta in the range [0.1, 2] Rong 2019   0.30  0.018230462799259543
kz = kc*(1/(ze*ze*ze))
zs = 1/(2*ze*ze)

# start measuring cpu training time

st = time.process_time()

# training 

epoch = 2

epo = 0

mc = 0; vc = 0                      # first moment and second moment initialzation

ms = 0; vs = 0

alpha = 0.13; eps = 0.00000001      # step size and epslon; set alpha = 0 if gradient is not needed

beta1 = 0.090; beta2 = 0.999999     # exponential decay rates  0.090 0.999

while epo < epoch:

#    print('epo = ', epo)

    for t in range(ndatatrain):
        x = Xtrain[t]
        d = gaussgranules(x, c, s, nc, ni)
        sumd = sum(d)
        for j in range(r):
            a[2*j] = d[j]*d[j]/sumd
            a[2*j+1] = d[j]/sumd
        sq = (np.absolute(ytrain[t] - np.dot(a,u)))*(np.absolute(ytrain[t] - np.dot(a,u)))

#    psi = 1

        psi =  kz*np.exp(-(sq/zs))
        b = lamb + psi*np.dot(np.dot(a, P), a)
        P = (P - (psi*np.outer(np.dot(P,a), np.dot(a,P)))/b)/lamb
        u = u + psi*np.dot(P,a)*(ytrain[t] - np.dot(a,u))   # model parameters
        o = np.dot(a,u)                                     # model output
        OT[t] = o
        norm[t] = la.norm(u)

        e = o - ytrain[t]

        if epo < epoch - 1:           # gradient steps

            for j in range(r):
                for i in range(ni):
                    gradc = 2*e*(d[j]/sumd)*((a[2*j]*u[2*j] + a[2*j+1]*u[2*j+1]) - o)*(x[i] - c[j][i])/(s[j][i]**2)
                    grads = gradc*(x[i] - c[j][i])/s[j][i]
                    mc = beta1*mc + (1 - beta1)*gradc
                    vc = beta2*vc + (1 - beta2)*gradc*gradc
                    mchat = mc/(1 - beta1)
                    vchat = vc/(1 - beta2)
                    c[j][i] = c[j][i] - alpha*mchat/(np.sqrt(vchat) + eps)        

                    ms = beta1*ms + (1 - beta1)*grads
                    vs = beta2*vs + (1 - beta2)*grads*grads
                    mshat = ms/(1 - beta1)
                    vshat = vs/(1 - beta2)
                    s[j][i] = s[j][i] - alpha*mshat/(np.sqrt(vshat) + eps)         
            
    epo = epo + 1

et = time.process_time()

# get execution time training

res = et - st

print('cpu execution time training:', res, 'seconds')

plt.figure()
plt.plot(norm, label="norm")
plt.xlabel('steps')
plt.ylabel('Norm')
plt.title('Norm')
plt.show()

# testing 

for t in range(ndatatest):
    x = Xtest[t]
    d = gaussgranules(x, c, s, nc, ni)
    sumd = sum(d)
    for j in range(r):
        a[2*j] = d[j]*d[j]/sumd
        a[2*j+1] = d[j]/sumd

    b = lamb + np.dot(np.dot(a, P), a)
    P = (P - (np.outer(np.dot(P,a), np.dot(a,P)))/b)/lamb    
    u = u + np.dot(P,a)*(ytest[t] - np.dot(a,u))    # model parameters recurice
    
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
plt.title('Prediction (train)')
#plt.ylim([0, 2])


print('MSE   test =', mean_squared_error(ytest, OS))
print('NRMSE test =', root_mean_squared_error(ytest, OS)/ytest.std())
print('RMSE  test =', root_mean_squared_error(ytest, OS))
print('MAE  test  =', mean_absolute_error(ytest, OS))
    

plt.figure()
plt.plot(ytest, label="Actual")
plt.plot(OS, label="ALSM")
plt.legend(loc='upper right')
plt.xlabel('k')
plt.ylabel('y')
plt.title('Prediction (test)')
#plt.ylim([0, 2])

plt.show()
