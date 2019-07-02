import numpy as np
import math
from autograd import elementwise_grad as egrad
from dijkstra import dijkstra
from stored_graph import option

N, G, OD, MaxStep = option("incomplete_5_10")


CCost = 0.1
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



"""
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
"""

"""
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
"""

def minCost():
    result = 0
    for i in range(N):
        for j in range(N):
            cost = G[i, j, 0] / G[i, j, 2]
            if result == 0 or (cost != 0 and cost < result):
                result = cost
    return result

def cost(P):
    curOD = OD 
    cost = 0
    for i in range(MaxStep - 1):
        Ni = np.tensordot(curOD, P[i], axes = ([1], [1])).diagonal().transpose()
        for j in range(N):
            for k in range(N):
                if G[j,k,0] == 0:
                    cost += CCost
                else:
                    cost += (G[j,k,0]/ (G[j,k,2] * f(Ni[j,k]/G[j,k,1])))*Ni[j,k]
        curOD = np.tensordot(curOD, P[i], axes= ([0],[0])).diagonal()

    
    for i in range(N):
        for j in range(N):
            cost += (G[i,j,0]/ (G[i,j,2] * f(curOD[i,j]/G[i,j,1])))*curOD[i,j]


    return cost

def differentiate(P):
    return egrad(cost)(P)

def shortestPaths():
    def shortestPath(i,j):
        pathLen = 1
        graphCon = np.ones((N,N))
        for a in range(N):
            for b in range(N):
                if G[a,b,0] == -1:
                    graphCon[a,b] = 0
        while graphCon[i,j] == 0:
            pathLen += 1
            graphCon = np.matmul(graphCon,graphCon)
        return pathLen
    pathLenG = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i != j:
                pathLenG[i,j] = shortestPath(i,j)
    return pathLenG

def findP():
    pathLenG = shortestPaths()
    graphCon = np.ones((N,N))
    for a in range(N):
        for b in range(N):
            if G[a,b,0] == -1:
                graphCon[a,b] = 0
    P = np.zeros((MaxStep-1, N,N,N))
    for s in range(MaxStep - 1):
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    if (graphCon[i,k] == 1 and pathLenG[k,j] < pathLenG[i,j]) or (pathLenG[i,j] == 0 and i == k):
                        P[s,i,j,k] = 1 
                        break 
    return P
def naiveCost():
    minTimeG = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if G[i,j,0] >= 0:
                minTimeG[i,j] = G[i,j,0]/G[i,j,2]
            else:
                minTimeG[i,j] = -1
    shortestRouteG = np.zeros((N,N))
    paths = []

    for i in range(N):
        curNm = np.zeros((N,N))
        curD, path = dijkstra(minTimeG, i)
        paths.append(path)
        shortestRouteG[i] = curD
    notComplete = True
    index = 0
    allN = []
    while notComplete:
        notComplete = False
        curN = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                curp = paths[i][j]
                if len(curp) > index:
                    notComplete = True
                    if index == 0:
                        last = i
                    else:
                        last = curp[index - 1]
                    now = curp[index]
                    curN[last,now] += OD[i,j]
        index += 1
        if notComplete:
            allN.append(curN)
    curCost = 0
    for matN in allN:
        for i in range(N):
            for j in range(N):
                if matN[i,j]!= 0:
                    curCost += matN[i,j] * G[i,j,0] / ((G[i,j,2] * f(matN[i,j]/G[i,j,1])))
    return curCost

#print(minCost()/MaxStep)

