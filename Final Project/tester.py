import numpy as np

import edge_matrix
import sql_map

from constants import *


def test():
    # read the edge matrix
    edge_mat = edge_matrix.EdgeMatrix()

    # For example print the second articles in citations
    print(edge_mat[1])

    # sql map hash to index
    id_to_index_map = sql_map.SqlMap("HASH_TO_INDEX", HASH_TO_INDEX_PATH)

    # print index of test id
    print(id_to_index_map[TEST_ID])

    # sql map hash to index
    index_to_hash_map = sql_map.SqlMap("INDEX_TO_HASH", INDEX_TO_HASH_PATH)

    # print index of test id
    print(index_to_hash_map[5])

    # out citations
    out_citations = np.memmap(OUT_CITATIONS_PATH, dtype="int64", mode="r+", shape=(10**8 * 4))

    # test out citations
    for i in range(100):
        print(out_citations[3000000 + i])

test()




