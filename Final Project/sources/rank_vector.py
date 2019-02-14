import json
import sqlite3
import time
import shelve
import os

from constants import *


VEC_TYPES = [
    '''
        CREATE TABLE "{}"
        (ID TEXT PRIMARY KEY NOT NULL,
        num TEXT NOT NULL);
    ''',
    '''
        CREATE TABLE "{}"
        (ID INT PRIMARY KEY NOT NULL,
        num TEXT NOT NULL);
    '''
]


class Vector():
    def __init__(self, name, type, url_path):
        self._name = name
        self._conn = sqlite3.connect(url_path)
        self._cur = self._conn.cursor()

        # check if table exists, if not create TABLE
        self._cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = self._cur.fetchall()
        if name not in [val[0] for val in tables]:
            self._conn.execute(VEC_TYPES[type].format(self._name.replace('"', '""')))


    def __setitem__(self, index, edges):
        try:
            self._conn.execute(
                """
                INSERT INTO "{}" (ID, num) 
                VALUES (?, ?);
                """.format(self._name.replace('"', '""')), (index, edges)
            )
        except:
            print("Update Failed")

    def __getitem__(self, index):
        self._cur.execute(
            """
            SELECT * FROM "{}"
            WHERE ID = ?;
            """.format(self._name.replace('"', '""')), (index,)
        )
        try:
            return self._cur.fetchall()[0][1]
        except Exception as e:
            print(e)
            return None

    def get_multiple(self, keys):
        print(keys)
        if len(keys) == 0:
            return []

        keys = [(key,) for key in keys]
        print(keys)
        self._cur.executemany(
            """
            SELECT * FROM "{}"
            WHERE ID = ?;
            """.format(self._name.replace('"', '""')), keys
        )
        try:
            a = [val[1] for val in self._cur.fetchall()]
            print(a)
            return a
        except Exception as e:
            print(e)
            return []

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
