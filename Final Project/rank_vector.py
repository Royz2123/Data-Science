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
        try:
            self._conn.execute(VEC_TYPES[type].format(self._name.replace('"', '""')))
        except:
            pass
        self._cur = self._conn.cursor()

    def __setitem__(self, index, edges):
        self._conn.execute(
            """
            INSERT INTO "{}" (ID, num) 
            VALUES (?, ?);
            """.format(self._name.replace('"', '""')), (index, edges)
        )

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
