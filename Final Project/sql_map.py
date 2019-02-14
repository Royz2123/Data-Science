import json
import sqlite3
import time
import shelve
import os

from constants import *

# SQL Stuff

LIST_TABLES = """
    SELECT name FROM sqlite_master WHERE type='table';
"""
CREATE_TABLE = '''
    CREATE TABLE "{}"
    (ID TEXT PRIMARY KEY NOT NULL,
    num TEXT NOT NULL);
'''
INSERT = """
    INSERT INTO "{}" (ID, num) 
    VALUES (?, ?);
"""
SELECT = """
    SELECT * FROM "{}"
    WHERE ID = ?;
"""


class SqlMap(object):
    def __init__(self, name, url_path):
        self._name = name
        self._conn = sqlite3.connect(url_path)
        self._cur = self._conn.cursor()

        # check if table exists, if not create TABLE
        self._cur.execute(LIST_TABLES)
        tables = self._cur.fetchall()
        if name not in [val[0] for val in tables]:
            self._conn.execute(CREATE_TABLE.format(self._name.replace('"', '""')))

    def __setitem__(self, index, edges):
        try:
            self._conn.execute(INSERT.format(self._name.replace('"', '""')), (index, edges))
        except:
            print("Update Failed")

    def __getitem__(self, index):
        self._cur.execute(SELECT.format(self._name.replace('"', '""')), (index,))
        try:
            return self._cur.fetchall()[0][1]
        except Exception as e:
            print(e)
            return None

    def save(self):
        self._conn.commit()

    def close(self):
        self._conn.close()

"""
vec = Vector("yoav_table", 0, EDGES_VECTOR_PATH)
print(vec[0])
vec[0] = "yo"
print(vec[0])
vec.save()

"""
