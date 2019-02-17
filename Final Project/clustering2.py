import heapq
import json

import nltk as nltk
import numpy as np
import time

import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from scipy.cluster.hierarchy import ward, dendrogram
from scipy.cluster.hierarchy import cut_tree

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from constants import *

import edge_matrix
import sql_map

from constants import *
from sources.rank_vector import Vector

RECORDS = int(10 ** 7 * 3.918)
articlesNum = 200


def getTextDatasFromDatabase():
    print("getting saved text data...")
    vec = Vector("text_data_vector", 1, BEST_TEXT_DATA_VEC)
    if vec is None:
        return []
    l = []
    for i in range(articlesNum):
        a = vec[i]
        a = a.split("$")
        a[1] = a[1].split("@")
        l.append(a)
    return l

def createTextDataDatabse(indexes):
    print("creating text vector...")
    vec = Vector("text_data_vector", 1, BEST_TEXT_DATA_VEC)
    j = 0
    for i in indexes:
        print(j)
        data = getTextDataFromPaper(i)
        data[1] = "@".join(data[1])
        vec[j] = "$".join(data)
        j += 1
    vec.save()
    print("done.")


def getTextDataFromPaper(index):
    pathname = RAW_DATA_PATH + str(index // 1000000)
    with open(pathname, 'rb') as f:
        index = index % 1000000
        for line in f:
            data = json.loads(line)
            if index == 0:
                return [data["title"], data["entities"], data["paperAbstract"]]
            index -= 1


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

def createBestArticles():
    print("finding best articles...")
    open("databases/1000BestVector", "w+").close()
    bestVector = np.memmap("databases/1000BestVector", dtype="int64", mode="r+", shape=(articlesNum))
    best = calcBestArticles(articlesNum)

    for i in range(articlesNum):
        bestVector[i] = best[i][1]
        print(best[i])

    bestVector.flush()

def getBestArticlesIndexes():
    print("getting best articles indexes...")
    indexes = []
    bestVector = np.memmap("databases/1000BestVector", dtype="int64", mode="r+", shape=(articlesNum))
    for i in range(articlesNum):
        indexes.append(bestVector[i])
    return indexes

def augmented_dendrogram(*args, **kwargs):

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        a = 1
        c = {}
        leafPos = []
        y = []
        for d in ddata['dcoord']:
            y.extend(d)
        leafY = min(y)
        for i, d in zip(ddata['icoord'], ddata['dcoord']):
            for x,y in zip(i,d):
                if (y == leafY):
                    leafPos.append(x)
        leafPos = sorted(set(leafPos))
        print(len(leafPos))
        for i, d in zip(ddata['icoord'], ddata['dcoord']):
            a += 1
            x = 0.5 * sum(i[1:3])
            y = d[1]
            plt.plot(x, y, 'ro')
            plt.annotate(str(a), (x, y), xytext=(0, -8),
                         textcoords='offset points',
                         va='top', ha='center')
            # plt.annotate("%.3g" % y, (x, y), xytext=(0, -8),
            #              textcoords='offset points',
            #              va='top

    return ddata

# createBestArticles()
indexes = getBestArticlesIndexes()
newIndexes = indexes.copy()
# createTextDataDatabse(indexes)
textDatas = getTextDatasFromDatabase()
newData = []
abstracts = []
titles = []
for i in range(articlesNum):
    data = textDatas[i]
    if (len(data[2].split(" ")) > 7): #if the abstract exists and is not too small
        newIndexes.append(indexes[i])
        newData.append(data)
        abstracts.append(textDatas[i][2])
        titles.append(textDatas[i][0])
        print(i, indexes[i], data)
indexes = newIndexes
textDatas = newData


STOPWORDS = stopwords.words("english")
stemmer = SnowballStemmer("english")

print("starting clustering...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=0.05,
                                 stop_words='english', ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(abstracts)
print("clustering ended")
terms = tfidf_vectorizer.get_feature_names()
print(terms)
dist = 1 - cosine_similarity(tfidf_matrix)

linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

#
# from scipy.cluster.hierarchy import fcluster
# tree = cut_tree(linkage_matrix)
#
#
# fig, ax = plt.subplots(figsize=(30, 50)) # set size
# # ax = augmented_dendrogram(linkage_matrix, orientation="right", labels=titles)
# augmented_dendrogram(linkage_matrix,
#                # color_threshold=1,
#                # p=6,
#                # truncate_mode='lastp',
#                 show_leaf_counts=True,
#                )
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False)
#
# plt.tight_layout() #show plot with tight layout
# plt.show()
#uncomment below to save figure
# plt.savefig('ward_clusters.png', dpi=200)