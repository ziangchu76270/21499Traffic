import numpy as np 
from quadprograming import optimization
from autodif import N, G, MaxStep, differentiate, differentiate_2, cost, shortestPaths, findP, naiveCost

optimalP = np.round(optimization(findP()),decimals = 2) 
print(optimalP)
print(cost(optimalP))
print(naiveCost())
