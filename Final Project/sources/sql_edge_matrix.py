import json
import sqlite3
import time
import shelve
import os

from rank_vector import Vector
from constants import *


class EdgeMatrix(object):
    def __init__(self, init=True):
        self._edges_vec = Vector("EDGES", 0, EDGES_VECTOR_PATH)
        print(self._edges_vec[0])
        if init:
            self.init_vectors()

    def get_cited(self, index):
        if self._edges_vec[index] == "" or self._edges_vec[index] is None:
            return []
        return [int(val) for val in self._edges_vec[index].split(",")]

    def close(self):
        self._edges_vec.close()

    def init_vectors(self):
        edge_index = 0
        indexes = Vector("INDEXES", 1, ID_TO_INDEX_PATH)
        start_time = time.time()

        for filename in sorted(os.listdir(RAW_DATA_PATH), key=lambda x: int(x)):
            pathname = RAW_DATA_PATH + filename
            print(pathname)

            with open(pathname, 'rb') as f:
                for line in f:
                    data = json.loads(line)

                    citations = []
                    for citation in data["inCitations"]:
                        try:
                            citations.append(indexes[citation])
                        except:
                            pass
                    #citations = indexes.get_multiple(data["inCitations"])


                    self._edges_vec[edge_index] = ",".join([str(i) for i in citations])
                    edge_index += 1
                    if not edge_index % 1000:
                        print(edge_index, "\tFinished. Elapsed time: %.2f sec" % (time.time() - start_time))
                        self._edges_vec.save()
        self._edges_vec.save()
