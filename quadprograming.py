import numpy as np
from qpsolvers import solve_qp
from autodif import N, G, MaxStep, differentiate, cost, shortestPaths, findP
P = findP()
pathLenG = shortestPaths()
print(pathLenG)
def qp(P):

	C = differentiate(P)
	P_new = np.zeros((MaxStep-1,N,N,N))
	for k in range(len(P)):
		for i in range(N):
			for j in range(N):
				connected_roads = []
				q = []
				for l in range(N):
					if G[i,l,0] != -1 and (MaxStep - 1 - k >= pathLenG[l,j]):
						connected_roads.append(l)
						q.append(C[k,i,j,l] - P[k,i,j,l])
				q = np.asarray(q)
				n = len(connected_roads)
				M = np.identity(n)
				A = np.ones(n)
				b = np.array([1])
				I = -np.identity(n)
				h = np.zeros(n)
				raw_sol = solve_qp(M,q,I,h,A,b)
				sol = []
				for t in range(N):
					if t in connected_roads:
						sol.append(raw_sol[connected_roads.index(t)])
					else:
						sol.append(0)
				P_new[k,i,j] = sol
	return P_new


def optimization(P):
	def find_stepsize(P,d):
		alpha = 1
		tau = 0.9
		theta = 0.5
		dif = np.sum(np.multiply(d,differentiate(P)))
		step = 0
		costp = cost(P)
		while cost(P + alpha*d) >= costp + alpha*dif*tau and step < 15:
			step += 1
			alpha *= theta
		return alpha

	P_new = np.zeros((MaxStep - 1, N,N,N))
	step = 0
	d = np.ones(N)
	while np.max(d) > 0.01 and step < 250:
		print("d", step, np.max(d))
		step += 1
		#print(step)
		P_new = qp(P)
		d = P_new - P
		#print("asdfgh\n\n\n", P_new, P)
		step_size = find_stepsize(P,d)
		print(step_size)
		P = P + step_size*d 
	print(np.max(d))
	return P 


print(np.round(optimization(P),decimals = 2))

