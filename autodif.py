import numpy as np
import random
import autograd as grad
import math
from autograd import elementwise_grad as egrad
from stored_graph import option

N, G, OD = option("defaultx")


# plan for step1    
P = [[[0 for _ in range(N)] for _ in range(N)]for _ in range(N)]
for i in range(N):
    for j in range(N):
        for k in range(N):
            if j == k:
                P[i][j][k] = 1.0

P = np.asarray(P)
#print(P)

# cost function
def f(x): #x is number of cars per length unit per lane
    if x < 0:
        return -1
    if 0 <= x and x <= 1:
        return 1
    if x > 1:
        return 1/x

def shortestPaths(G):
  result = [[[] for _ in range(N)]for _ in range(N)]
  minCost = np.asarray([[9999999 for _ in range(N)] for _ in range(N)])
  for i in range(N):
    for j in range (N):
      for k in range (N):
        if (G[i, k, 0] / G[i, k, 2] + G[k, j, 0] / G[k, j, 2]) == minCost[i, j]:
            result[i][j].append(k)
        if (G[i, k, 0] / G[i, k, 2] + G[k, j, 0] / G[k, j, 2]) < minCost[i, j]:
            result[i][j] = [k]
            minCost[i, j] = (G[i, k, 0] / G[i, k, 2] + G[k, j, 0] / G[k, j, 2])
  return result

def cost(P):
    # N2 = 0
    N1 = np.tensordot(OD, P, axes = ([1], [1])).diagonal().transpose()
    N2 = np.tensordot(OD, P, axes = ([0], [0])).diagonal()
    #print(N1)
    #current velocity
    cost = 0

    for i in range(N):
        for j in range(N):
            cost += G[i,j,0]/ (G[i,j,2] * f(N1[i,j]/G[i,j,1]))
            cost += G[i,j,0]/ (G[i,j,2] * f(N2[i,j]/G[i,j,1]))

    return cost 

def newCost():
    Q = [[[0 for _ in range(N)] for _ in range(N)]for _ in range(N)]
    Q = np.asarray(Q)
    ODX = OD * 0.6
    ODY = OD * 0.4
    paths = shortestPaths(G)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if ODY[i][j] > 0:
                    m = len(paths[i][j])
                    Q[i, j, k] = 1/m
    N1X = np.tensordot(ODX, P, axes = ([1], [1])).diagonal().transpose()
    N2X = np.tensordot(ODX, P, axes = ([0], [0])).diagonal()

    N1Y = np.tensordot(ODY, Q, axes = ([1], [1])).diagonal().transpose()
    N2Y = np.tensordot(ODY, Q, axes = ([0], [0])).diagonal()

    cost = 0

    for i in range(N):
        for j in range(N):
            cost += G[i,j,0]/ (G[i,j,2] * f((N1X[i,j] + N1Y[i, j])/G[i,j,1]))
            cost += G[i,j,0]/ (G[i,j,2] * f((N2X[i,j] + N2Y[i, j])/G[i,j,1]))

    return cost

def differentiate(P):
    return egrad(cost)(P)



# print(P)

# print(differentiate(P))


