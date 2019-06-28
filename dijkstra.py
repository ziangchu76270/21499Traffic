def pop_smallest(pq):
    smallest_dis = None
    smallest_node = 0
    index = 0
    for i in range(len(pq)):
        (dis, node) = pq[i]
        if smallest_dis is None or dis < smallest_dis:
            smallest_dis = dis
            smallest_node = node
            index = i
    pq = pq[:index] + pq[index + 1:]
    return (smallest_dis, smallest_node), pq


def deleteall(pq, i):
    index = 0
    while index < len(pq):
        item = pq[index]
        if item[1] == i:
            pq.pop(index)
        else:
            index += 1
    return pq


def dijkstra(graph, node):
    pq = []
    pq.append((0, node))
    alldistance = [None] * (len(graph))
    alldistance[node] = 0
    while pq != []:
        (dis, n), pq = pop_smallest(pq)
        for i in range(len(graph[n])):
            if graph[n][i] > 0 and (
                    alldistance[i] is None or alldistance[i] > alldistance[n] + graph[n][i]):
                alldistance[i] = alldistance[n] + graph[n][i]
                pq = deleteall(pq, i)
                pq.append((alldistance[i], i))
    return alldistance



