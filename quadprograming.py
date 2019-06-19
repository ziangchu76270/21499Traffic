import numpy as np
from qpsolvers import solve_qp
from autodif import N, P, differentiate, cost



def qp(P):
	C = differentiate(P)
	P_new = np.zeros((N,N,N))
	for i in range(N):
		for j in range(N):
			q = C[i,j] - P[i,j]
			M = np.identity(N)
			A = np.ones(N)
			b = np.array([1])
			G = -np.identity(N)
			h = np.zeros(N)
			P_new[i,j] = solve_qp(M,q,G,h,A,b)
	return P_new


def optimization(P):
	def find_stepsize(P,d):
		alpha = 1
		tau = 0.5
		theta = 0.5
		dif = np.sum(np.multiply(d,differentiate(P)))
		step = 0
		costp = cost(P)
		while cost(P + alpha*d) >= costp + alpha*dif*tau and step < 1000:
			step += 1
			alpha *= theta
		return alpha

	P_new = np.zeros((N,N,N))
	step = 0
	step_size = 1
	d = np.ones(N)
	while np.max(d) > 0.000001 and step < 100:
		print("d", np.max(d))
		#print(np.max(step_size*d))
		step += 1
		#print(step)
		P_new = qp(P)
		d = P_new - P
		step_size = find_stepsize(P,d)
		P = P + step_size*d 
	return P 


print(np.round(optimization(P),decimals = 8))

