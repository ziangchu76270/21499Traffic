import numpy as np
import random
import autograd as grad
import math
from autograd import elementwise_grad as egrad
from stored_graph import option

N, G, OD, MaxStep = option("manhattan")

# plan for step1    
unitP = [[[0 for _ in range(N)] for _ in range(N)]for _ in range(N)]
for i in range(N):
    for j in range(N):
        for k in range(N):
            if j == k:
                unitP[i][j][k] = 1.0

P = np.zeros((MaxStep-1, N,N,N))
for i in range(len(P)):
    P[i] = unitP

"""
# cost function
def f(x): #x is number of cars per length unit per lane
    if x < 0:
        return -1
    if 0 <= x and x <= 1:
        return 1
    if x > 1:
        return 1/x
"""

def f(x): 
    if x<0:
        return -1
    if 0<=x and x<=4.0/3:
        return -(0.75**3)/4*x**3+1
    if x > 4.0/3:
        return 1.0/x




def shortestPaths(G):
    result = [[[] for _ in range(N)]for _ in range(N)]
    minCost = np.asarray([[9999999 for _ in range(N)] for _ in range(N)])
    for i in range(N):
        for j in range (N):
            for k in range (N):
                if (G[i, k, 0] / G[i, k, 2] + G[k, j, 0] / G[k, j, 2]) < minCost[i, j]:
                    result[i][j] = [k]
                    minCost[i, j] = (G[i, k, 0] / G[i, k, 2] + G[k, j, 0] / G[k, j, 2])
                # TAI QIANG LE
                elif (G[i, k, 0] / G[i, k, 2] + G[k, j, 0] / G[k, j, 2]) == minCost[i, j]:
                    result[i][j].append(k)
         
    return result



def cost(P):
    curOD = OD 
    cost = 0
    for i in range(MaxStep - 1):
        Ni = np.tensordot(curOD, P[i], axes = ([1], [1])).diagonal().transpose()
        for j in range(N):
            for k in range(N):
                cost += (G[j,k,0]/ (G[j,k,2] * f(Ni[j,k]/G[j,k,1])))*Ni[j,k]
        curOD = np.tensordot(curOD, P[i], axes= ([0],[0])).diagonal()

    
    for i in range(N):
        for j in range(N):
            cost += (G[i,j,0]/ (G[i,j,2] * f(curOD[i,j]/G[i,j,1])))*curOD[i,j]


    return cost 

def differentiate(P):
    return egrad(cost)(P)


print(P)

print(differentiate(P))


