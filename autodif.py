import numpy as np
import random
import autograd as grad
import math
from autograd import elementwise_grad as egrad

# number of nodes in the graph
N = 4

G = np.asarray([[[0, -1, -1], #from 1 to 1, distance, number of lanes, v_m
          [10, 4, 5],
          [12, 2, 8],
          [20, 5, 5]],

         [[10, 2, 5],
          [0, -1, -1],
          [6, 2, 10],
          [7, 5, 8.5]],

         [[12, 2, 8],
          [6, 2, 10],
          [0, -1, -1],
          [18, 4, 10]],

         [[20, 5, 5],
          [7, 5, 8.5],
          [18, 4, 10],
          [0, -1, -1]]])
  
# OD matrix  
OD = np.asarray([[4, 4, 4, 3], 
         [5, 2, 6, 6],
         [4, 7, 5, 1],
         [9, 8, 0, 6]])


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
        if (G[i, k, 0] / G[i, k, 2] + G[k, j, 0] / G[k, j, 2]) <= minCost[i, j]:
          result[i][j].append(k)
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

def differentiate(P):
    return egrad(cost)(P)



# print(P)

# print(differentiate(P))


