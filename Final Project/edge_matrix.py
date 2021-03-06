import json
import sqlite3
import time
import shelve
import os
import numpy as np
import sql_map

from constants import *


class EdgeMatrix(object):
    BASE_FOLDER = "databases/"
    OFFSETS_FILENAME = "Offset_Vector"
    EDGES_FILENAME = "Edges_Vector"
    OFFSETS_PATH = BASE_FOLDER + OFFSETS_FILENAME
    EDGES_PATH = BASE_FOLDER + EDGES_FILENAME

    def __init__(self, create_new=False):
        if create_new or EdgeMatrix.OFFSETS_FILENAME not in os.listdir(EdgeMatrix.BASE_FOLDER):
            open("databases/Offset_Vector", 'w+').close()
            open("databases/Edges_Vector", 'w+').close()

            self._offset_vec = np.memmap(EdgeMatrix.OFFSETS_PATH, dtype="int64", mode="r+", shape=(10**8 * 4))
            self._edges_vec = np.memmap(EdgeMatrix.EDGES_PATH, dtype="int64", mode="r+", shape=(10**8 * 5))

            self.init_vectors()
        else:
            self._offset_vec = np.memmap(EdgeMatrix.OFFSETS_PATH, dtype="int64", mode="r+", shape=(10 ** 8 * 4))
            self._edges_vec = np.memmap(EdgeMatrix.EDGES_PATH, dtype="int64", mode="r+", shape=(10 ** 8 * 5))

    def __getitem__(self, index):
        start_offset = self._offset_vec[index]
        next_offset = self._offset_vec[index + 1]
        return [self._edges_vec[offset] for offset in range(start_offset, next_offset)]

    def init_vectors(self):
        edge_index = 0
        offset_index = 0
        conn = sqlite3.connect(ID_TO_INDEX_PATH)
        cur = conn.cursor()
        start_time = time.time()

        for filename in sorted(os.listdir(RAW_DATA_PATH), key=lambda x: int(x)):
            pathname = RAW_DATA_PATH + filename
            print(pathname)

            with open(pathname, 'rb') as f:
                for line in f:
                    data = json.loads(line)
                    self._offset_vec[edge_index] = offset_index

                    for citation in data["inCitations"]:
                        cur.execute(
                            """
                            SELECT * FROM INDEXES
                            WHERE ID = ?;
                            """, (citation,)
                        )
                        try:
                            cit_index = cur.fetchall()[0][1]
                            self._edges_vec[offset_index] = cit_index
                            offset_index += 1
                        except Exception as e:
                            print(e)

                    edge_index += 1

                    if not edge_index % 10000:
                        print(edge_index, "\tFinished. Elapsed time: %.2f sec" % (time.time() - start_time))
                        self._edges_vec.flush()
                        self._offset_vec.flush()
        self._edges_vec.flush()
        self._offset_vec.flush()

