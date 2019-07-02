import numpy as np 
from quadprograming import optimization
from autodif import N, G, MaxStep, differentiate, cost, shortestPaths, findP, naiveCost, wait_cost

optimalP = np.round(optimization(findP()),decimals = 2) 
print(optimalP)
print(cost(optimalP))
print(wait_cost(optimalP))
print(naiveCost())

