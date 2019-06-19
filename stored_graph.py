import numpy as np 

def option(s):
    if s == "default":
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
        PROP = 1
    else:
        N = 4
        G = np.asarray([[[0, -1, -1], #from 1 to 1, distance, number of lanes, v_m
                [5, 2, 3],
                [5, 2, 3],
                [11, 2.0, 3]],

               [[5, 2, 3],
                [0, -1, -1],
                [5, 2, 3],
                [5, 2, 3]],

               [[5, 2, 3],
                [5, 2, 3],
                [0, -1, -1],
                [5, 2, 3]],

               [[5, 2, 3],
                [5, 2, 3],
                [5, 2, 3],
                [0, -1, -1]]])
        
        # OD matrix  
        OD = np.asarray(
              [[0, 0, 0, 1000000000.0], 
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]])
        PROP = 1
    return N, G, OD, PROP
