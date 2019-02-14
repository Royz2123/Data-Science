# each JSON is small, there's no need in iterative processing
import json
import sqlite3
import time
import shelve
import os
import numpy as np

import sql_map
from constants import *


def store_raw_data():
    index = 0
    hash_to_index = sql_map.SqlMap("HASH_TO_INDEX", HASH_TO_INDEX_PATH)
    index_to_hash = sql_map.SqlMap("INDEX_TO_HASH", INDEX_TO_HASH_PATH)
    start_time = time.time()

    for filename in sorted(os.listdir(RAW_DATA_PATH), key=lambda x: int(x)):
        pathname = RAW_DATA_PATH + filename
        print(pathname)

        with open(pathname, 'rb') as f:
            for line in f:
                data = json.loads(line)
                hash_to_index[data["id"]] = index
                index_to_hash[index] = data["id"]
                index += 1

                if not index % 1000:
                    print(index, "\tFinished. Elapsed time: %.2f sec" % (time.time() - start_time))
                    hash_to_index.save()
                    index_to_hash.save()
    hash_to_index.save()
    index_to_hash.save()
    hash_to_index.close()
    index_to_hash.close()


def store_raw_data_out():
    index = 0
    out_citations = np.memmap(OUT_CITATIONS_PATH, dtype="int64", mode="r+", shape=(10**8 * 4))
    start_time = time.time()

    for filename in sorted(os.listdir(RAW_DATA_PATH), key=lambda x: int(x)):
        pathname = RAW_DATA_PATH + filename
        print(pathname)

        with open(pathname, 'rb') as f:
            for line in f:
                data = json.loads(line)
                out_citations[index] = len(data["outCitations"])
                index += 1

                if not index % 1000:
                    print(index, "\tFinished. Elapsed time: %.2f sec" % (time.time() - start_time))
                    out_citations.flush()
    out_citations.flush()


def store_raw_data_old():
    index = 0
    indexes = sql_map.SqlMap("INDEXES_TEST", ID_TO_INDEX_PATH)
    start_time = time.time()

    for filename in sorted(os.listdir(RAW_DATA_PATH), key=lambda x: int(x)):
        pathname = RAW_DATA_PATH + filename
        print(pathname)

        with open(pathname, 'rb') as f:
            for line in f:
                data = json.loads(line)
                indexes[data["id"]] = index
                index += 1

                if not index % 1000:
                    print(index, "\tFinished. Elapsed time: %.2f sec" % (time.time() - start_time))
                    indexes.save()
    indexes.close()


# Create all databases
def init_databases():
    store_raw_data()
    store_raw_data_out()

init_databases()























