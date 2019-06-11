import numpy as np
import random
import autograd as grad
import math


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
P = [ [[0 for _ in range(N)] for _ in range(N)]for _ in range(N)]
for i in range(N):
    for j in range(N):
        for k in range(N):
            if j == k:
                P[i][j][k] = 1

P = np.asarray(P)
print(P)

# cost function
def f(x): #x is number of cars per length unit per lane
    if x < 0:
        return -1
    if 0 <= x and x <= 1:
        return 1
    if x > 1:
        return 1/x


def cost(P):
    T = 0 # total time
    V = 0 # average velocity
    # N2 = 0
    # for i in range(0, n): 
    #      N2_kj += P[i][j][k] * x[i][j][k]
    N1 = np.tensordot(OD,P, axes=([1],[1])).diagonal().transpose()
    print(N1)
    #current velocity
    cost = 0


    for i in range(N):
        for j in range(N):
            cost += G[i,j,0]/ (G[i,j,2] * f(N1[i,j]/G[i,j,1]))
            print(cost)
    return cost 
"""
    v = G[:,:,2]*f(N1[:,:]/G[:,:,1])
  
    #Total Time
    T= np.zeros(N)
    T= np.sum(G[:,:0]/v)
    
    #Average Velocity
    V = np.zeros(N)
    V = np.sum(v*N1)
    # N1 = np.tensordot(P, OD, axes = 1)
   
    # N1 = [[0, 0, 0, 0], # ppl from i to k in P1
    #       [0, 0, 0, 0],
    #       [0, 0, 0, 0],
    #       [0, 0, 0, 0]]

    # for i in range (0, n):
    #     for k in range (0, kmax):
    #         for j in range (0, jmax):
    #             N1[i][k] += P[i][j][k] * OD[i][j]

    # v = list(range(4)) #current speed of each road (2d matrix)
    # for i in range (0, 4):
    #     v[i] = list(range(4))
    
    # for i in range (0, n):
    #     for k in range (0, kmax):
    #         if (G[i][k][0] != -1):
    #             v[i][k] = G[i][k][2] * f(N1[i][k] / (G[i][k][0] * G[i][k][1]))

    # for i in range (0, n): #T is the total travel time of step1
    #     for k in range (0, kmax):
    #         if (G[i][k][0] != -1):
    #             T += G[i][k][0] / v[i][k]

    # for i in range (0, n): #V: total speed
    #     for k in range (0, kmax):
    #         V += v[i][k] * N1[i][k]
"""


def autodif_init(p):
    result = list(range(2))
    result[0] = p
    result[1] = []
    return result


class autodif(object):
    def __init__(self):
        self.input = [None]

def run(value, n):
    class Struct(object):pass
    data  = Struct()
    data.value = value
    data.diffs = []

def add(a, b):
    result = autodif_init(a + b)
    for i in range (0, n):
        for j in range (0, jmax):
            for k in range (0, kmax):
                result[1][i][j][k] = a[1][i][j][k] + b[1][i][j][k]
    return result

def mult(a, b):
    result = autodif_init(a * b)
    for i in range (0, n):
        for j in range (0, jmax):
            for k in range (0, kmax):
                result[1][i][j][k] = a[0][i][j][k] * b[1][i][j][k] + b[0][i][j][k] * a[1][i][j][k]
    return result


cost(P)

