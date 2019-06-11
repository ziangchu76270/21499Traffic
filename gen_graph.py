import numpy as np 
import random
from autograd import grad

# number of nodes in the graph
N = 4
DISTANCE = (5,20)
LINES = (1, 5)
VELOCITY = (3, 6)
DENSITY = 0.8
PPLTRAVEL = 10

def generate_graph(x, density):
	frontier = [0]
	num_edges = int((x-1)*density)
	step1 = zip(list(range(1,x)), [num_edges-1]*(x-1))

	step2 = []
	for _ in range(x - 1 - num_edges):
		k = random.randint(0, len(step1) - 1)
		step2.append((step1.pop(k)[0], num_edges))
	edges = []
	for node in step1:
		d = DISTANCE[0] + random.random() * DISTANCE[1]
		l = random.randint(LINES[0], LINES[1])
		v = VELOCITY[0] + random.random() * VELOCITY[1]
		edges.append((0, node[0], d, l, v))

	for i in range(len(step1)):
		node = step1[i]
		if node[1] > 0:
			for _ in range(node[1]):
				d = DISTANCE[0] + random.random() * DISTANCE[1]
				l = random.randint(LINES[0], LINES[1])
				v = VELOCITY[0] + random.random() * VELOCITY[1]
				sec_node_index = random.randint(i+1, len(step1)+len(step2) -1)
				if sec_node_index < len(step1):
					step1[sec_node_index] = (step1[sec_node_index][0], step1[sec_node_index][1]-1)
					sec_node = step1[sec_node_index][0]
				else:
					step2[sec_node_index-len(step1)] = (step2[sec_node_index-len(step1)][0],  
														step2[sec_node_index-len(step1)][1] - 1)
					sec_node = step2[sec_node_index - len(step1)][0]
				edges.append((node[0], sec_node, d, l, v))

	for i in range(len(step2) - 1):
		node = step2[i]
		if node[1] > 0:
			for _ in range(node[1]):
				d = DISTANCE[0] + random.random() * DISTANCE[1]
				l = random.randint(LINES[0], LINES[1])
				v = VELOCITY[0] + random.random() * VELOCITY[1]
				sec_node_index = random.randint(i+1, len(step2) -1)
				step2[sec_node_index] = (step2[sec_node_index][0], step2[sec_node_index][1] - 1)
				sec_node = step2[sec_node_index][0]
				edges.append((node[0], sec_node, d, l, v))

	graph = [[[-1,-1,-1] for _ in range(x)] for _ in range(x)]
	for edge in edges:
		graph[edge[0]][edge[1]] = [edge[2],edge[3],edge[4]]
	return np.asarray(graph)


def generate_od_matrix(x):
	od = [[0]*x for _ in range(x)]
	for i in range(x):
		for j in range(x):
			od[i][j] = random.randint(0,PPLTRAVEL)
	return np.asarray(od)
# Graph, distance, lines, Maximum velocity
G = generate_graph(N, DENSITY)
print(G)
# OD - matrix 
OD = generate_od_matrix(N)
print(OD)
# Portion of ppl traveling
P = []
