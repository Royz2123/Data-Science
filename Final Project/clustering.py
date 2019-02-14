import numpy as np
import time

import edge_matrix
import sql_map

from constants import *



def clustering_test():
    # read the edge matrix
    edge_mat = edge_matrix.EdgeMatrix()

    # Create numpy array of edges
    x = np.array([])
    x.astype(int)
    for i in range(39000):
        edges = [[int(i),int(neighbour)] for neighbour in edge_mat[i]]
        print(edges)
        time.sleep(1)
        x = np.append(x, np.array([edges]))
        print(x)

clustering_test()