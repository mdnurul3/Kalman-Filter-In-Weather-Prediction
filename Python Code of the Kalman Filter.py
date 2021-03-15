from numpy import dot
from numpy import dot, sum, tile, linalg
from numpy.linalg import inv,det 
from numpy import *
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('weather-raw_2020.csv')
df=df.fillna(0)
df.isnull().sum()
frames = [df['Temp'], df['Humidity (%)']]
result = pd.concat(frames, axis=1)
result.head(2)
Y_val=result.T.values
#Y.reshape((2, 1))
#Y=Y_val[:,0]
Y=Y_val[:,0].reshape((2,1))

def kf_predict(X, P, A, Q, B, U):
 X = dot(A, X) + dot(B, U)
 P = dot(A, dot(P, A.T)) + Q
 return(X,P) 

def kf_update(X, P, Y, H, R):
 IM = dot(H, X)
 IS = R + dot(H, dot(P, H.T))
 K = dot(P, dot(H.T, inv(IS)))
 X = X + dot(K, (Y-IM))
 P = P - dot(K, dot(IS, K.T))
 LH = gauss_pdf(Y, IM, IS)
 return (X,P,K,IM,IS,LH) 

def gauss_pdf(X, M, S):
 if M.shape[1] == 1:
  DX = X - tile(M, X.shape[1])
  E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
  E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(det(S))
  P = exp(-E)
 elif X.shape()[1] == 1:
  DX = tile(X, M.shape[1])- M
  E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
  E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(det(S))
  P = exp(-E)
 else: 
  DX = X-M
  E = 0.5 * dot(DX.T, dot(inv(S), DX))
  E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(det(S))
  P = exp(-E)
 return (P[0],E[0])


dt = 0.1
# Initialization of state matrices
X = array([[0.0], [0.0], [0.1], [0.1]])
P = diag((0.01, 0.01, 0.01, 0.01))
A = array([[1, 0, dt , 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0,\
 1]])
Q = np.eye(4, dtype=int)
B = np.eye(4, dtype=int)
U = zeros((X.shape[0],1))
H = array([[1, 0, 0, 0], [0, 1, 0, 0]])
R = eye(Y.shape[0])
# Number of iterations in Kalman Filter
N_iter = 50
x_pos=[]
y_pos=[]
# Applying the Kalman Filter
for i in range(N_iter):
 (X, P) = kf_predict(X, P, A, Q, B, U)
 (X, P, K, IM, IS, LH) = kf_update(X, P, Y, H, R)
 Y=Y_val[:,i].reshape((2,1))
x_pos.append(float(X[0]))
y_pos.append(float(Y[0]))
plt.plot(x_pos,y_pos)