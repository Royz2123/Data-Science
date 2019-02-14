# each JSON is small, there's no need in iterative processing
import json
import sqlite3
import time
import shelve
import os

from rank_vector import Vector
from constants import *


def store_raw_data():
    index = 0
    indexes = Vector("INDEXES_TEST", 1, ID_TO_INDEX_PATH)
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







store_raw_data()


conn = sqlite3.connect(ID_TO_INDEX_PATH)
























def create_hash_map():
    d = shelve.open(HASH_TO_INDEX_PATH)  # open -- file may get suffix added by low-level
                               # library

    key = "yo"
    data = "yoav"
    d[key] = data              # store data at key (overwrites old data if
                               # using an existing key)
    data = d[key]              # retrieve a COPY of data at key (raise KeyError
                               # if no such key)
    del d[key]                 # delete data stored at key (raises KeyError
                               # if no such key)

    flag = key in d            # true if the key exists
    klist = list(d.keys())     # a list of all existing keys (slow!)

    # as d was opened WITHOUT writeback=True, beware:
    d['xx'] = [0, 1, 2]        # this works as expected, but...
    d['xx'].append(3)          # *this doesn't!* -- d['xx'] is STILL [0, 1, 2]!

    # having opened d without writeback=True, you need to code carefully:
    temp = d['xx']             # extracts the copy
    temp.append(5)             # mutates the copy
    d['xx'] = temp             # stores the copy right back, to persist it

    # or, d=shelve.open(filename,writeback=True) would let you just code
    # d['xx'].append(5) and have it work as expected, BUT it would also
    # consume more memory and make the d.close() operation slower.

    d.close()                  # close it







create_hash_map()










"""
mydb = mysql.connector.connect(
  host="localhost",
  user="yourusername",
  passwd="yourpassword",
  database="mydatabase"
)

mycursor = mydb.cursor()

mycursor.execute("CREATE TABLE customers (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), address VARCHAR(255))")

"""