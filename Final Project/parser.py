# each JSON is small, there's no need in iterative processing
import json
import time
import mysql.connector
import shelve
import os
import pymongo

HASH_TO_INDEX_PATH = "databases/hash_to_index"
ADJACENCY_PATH = "databases/adjacency"
RAW_DATA_PATH = "raw_data/"


def store_raw_data():
    index = 0
    hash_to_index = shelve.open(HASH_TO_INDEX_PATH)  # open -- file may get suffix added by low-level
    start_time = time.time()

    for filename in sorted(os.listdir(RAW_DATA_PATH)):
        pathname = RAW_DATA_PATH + filename
        print(pathname)

        with open(pathname, 'rb') as f:
            for line in f:
                data = json.loads(line)
                hash_to_index[data["id"]] = index

                #print(data["title"])
                #print(hash_to_index[data["id"]])
                index += 1

                if not index % 1000:
                    print(index, "\tFinished. Elapsed time: %.2f sec" % (time.time() - start_time))
                    hash_to_index.sync()
    hash_to_index.close()



def store_raw_data2():
    index = 0
    hash_to_index = shelve.open(HASH_TO_INDEX_PATH)  # open -- file may get suffix added by low-level
    start_time = time.time()

    for filename in sorted(os.listdir(RAW_DATA_PATH)):
        pathname = RAW_DATA_PATH + filename
        print(pathname)

        with open(pathname, 'rb') as f:
            for line in f:
                data = json.loads(line)
                hash_to_index[data["id"]] = index

                #print(data["title"])
                #print(hash_to_index[data["id"]])
                index += 1

                if not index % 1000:
                    print(index, "\tFinished. Elapsed time: %.2f sec" % (time.time() - start_time))
                    hash_to_index.sync()
    hash_to_index.close()


def create_adjacency(filename):
    hash_to_index = shelve.open(HASH_TO_INDEX_PATH)

    with open(filename, 'rb') as raw_obj:
        with open(ADJACENCY_PATH, 'wb') as adj_obj:
            for line in raw_obj:
                data = json.loads(line)

                for cit_hash in data["outCitations"]:
                    # check if this id is in the database
                    try:
                        cit_index = hash_to_index[cit_hash]
                        adj_obj.write(", " + str(cit_index))

                    except Exception as e:
                        print(e)
                adj_obj.write("\n")

    hash_to_index.close()



store_raw_data()


























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
