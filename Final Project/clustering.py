import heapq
import json

import nltk as nltk
import numpy as np
import time

from nltk.corpus import stopwords

from constants import *

import edge_matrix
import sql_map

from constants import *
from sources.rank_vector import Vector

RECORDS = int(10 ** 7 * 3.918)
STOPWORDS = stopwords.words("english")


def getStemmedText(index):
    vec = Vector("stemmedText", 1, STEM_TXT_VEC)
    if vec is None:
        return []
    l = vec[index].split(",")
    return [(l[2*i][1:], l[2*i+1][:-1]) for i in range(len(l)//2)]

def createStemmedTextVectorForIndexes(indexes):
    print("creating stemmed text vector...")
    vec = Vector("stemmedText", 1, STEM_TXT_VEC)
    j = 0
    for i in indexes:
        print(j)
        j += 1
        vec[i] = ",". join([f"({w},{i})" for w, i in stemText(getAbstractFromPaper(i))])
    vec.save()
    print("done.")

def getAbstractFromPaper(index):
    pathname = RAW_DATA_PATH + str(index // 1000000)
    with open(pathname, 'rb') as f:
        index = index % 1000000
        for line in f:
            data = json.loads(line)
            if index == 0:
                return data["paperAbstract"] if data["paperAbstract"] != "" else " ".join(data["entities"])
            index -= 1

def stemText(txt):
    txt = nltk.word_tokenize(txt)
    txt = list(filter(lambda x: x.isalpha() and x.lower() not in STOPWORDS, txt))
    occ = [(word, txt.count(word)) for word in list(set(txt))]
    occ = sorted(occ, key=lambda x: x[1], reverse=True)
    return occ

def calcBestArticles(num):
    curr_vector = np.memmap("databases/RankVector10_6", dtype="float64", mode="r+", shape=(10 ** 7 * 4))
    heap = []
    for i in range(num):
        heapq.heappush(heap, (curr_vector[i], i))
    for i in range(num, RECORDS):
        curr_val = curr_vector[i]
        heapq.heappushpop(heap, (curr_val, i))
        if not i % 100000:
            print(i)

    return heap

articlesNum = 10
def createBestArticles():
    print("finding best articles...")
    open("databases/1000BestVector", "w+").close()
    bestVector = np.memmap("databases/1000BestVector", dtype="int64", mode="r+", shape=(articlesNum))
    best = calcBestArticles(articlesNum)

    for i in range(articlesNum):
        bestVector[i] = best[i][1]
        print(best[i])

    bestVector.flush()

def getBestArticles():
    print("getting best articles indexes...")
    indexes = []
    bestVector = np.memmap("databases/1000BestVector", dtype="int64", mode="r+", shape=(articlesNum))
    for i in range(articlesNum):
        indexes.append(bestVector[i])
    return indexes


createBestArticles()
createStemmedTextVectorForIndexes(getBestArticles())


# def clustering_test():
#     # read the edge matrix
#     edge_mat = edge_matrix.EdgeMatrix()
#
#     # Create numpy array of edges
#     x = np.array([])
#     x.astype(int)
#     for i in range(39000):
#         edges = [[int(i),int(neighbour)] for neighbour in edge_mat[i]]
#         print(edges)
#         time.sleep(1)
#         x = np.append(x, np.array([edges]))
#         print(x)

# clustering_test()