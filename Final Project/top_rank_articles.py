import rank_vector
import numpy as np
import json
import heapq

from constants import *


def calcBestArticles(num):
    curr_vector = np.memmap(FINAL_RANK_VEC, dtype="float64", mode="r+", shape=(RECORDS))
    heap = []
    for i in range(num):
        heapq.heappush(heap, (curr_vector[i], i))
    for i in range(num, RECORDS):
        curr_val = curr_vector[i]
        heapq.heappushpop(heap, (curr_val, i))
        if not i % 100000:
            print(i)

    return heap

def createBestArticlesIndexesVec():
    print("finding best articles...")
    open(BEST_INDEXES_VEC, "w+").close()
    bestVector = np.memmap(BEST_INDEXES_VEC, dtype="int64", mode="r+", shape=(BEST_ARTICLE_NUM))
    best = calcBestArticles(BEST_ARTICLE_NUM)

    for i in range(BEST_ARTICLE_NUM):
        bestVector[i] = best[i][1]

    bestVector.flush()


def getTextDataFromIndexes(indexes):
    original = indexes
    indexes = sorted(indexes)
    k = 0
    textData = {}
    while (len(indexes) > k):
        index = indexes[k]
        pathname = RAW_DATA_PATH + str(index // 1000000)
        with open(pathname, 'rb') as f:
            index = index % 1000000
            i = 0
            for line in f:
                data = json.loads(line)
                if index == i:
                    textData[indexes[k]] = ([data["title"], data["entities"], data["paperAbstract"]])
                    print(indexes[k])
                    if (len(indexes) > k + 1):
                        if indexes[k] // 1000000 != indexes[k + 1] // 1000000:
                            k += 1
                            break
                        index = indexes[k+1] % 1000000
                        k += 1

                    else:
                        return [textData[i] for i in original]
                i += 1

def createBestTextDataDatabse(indexes):
    """
    create a database that if article a is i'th best article, then vec[i] will be
    [title, entities, abstract] of a
    :param indexes:
    :return:
    """
    print("creating text vector...")
    vec = Vector(TEXT_DATA_VECTOR, 1, BEST_TEXT_DATA_VEC)
    datas = getTextDataFromIndexes(indexes)
    print(datas)
    j = 0
    for data in datas:
        print(j)
        data[1] = "@".join(data[1])
        vec[j] = "$".join(data)
        j += 1
    vec.save()
    print("done.")

def getTextDatasFromBestTxtVec():
    print("getting saved text data...")
    vec = rank_vector.Vector(TEXT_DATA_VECTOR, 1, BEST_TEXT_DATA_VEC)
    if vec is None:
        return []
    l = []
    for i in range(BEST_ARTICLE_NUM):
        a = vec[i]
        a = a.split("$")
        a[1] = a[1].split("@")
        l.append(a)
    return l


# def getTextDataFromPaper(index):
#     pathname = RAW_DATA_PATH + str(index // 1000000)
#     with open(pathname, 'rb') as f:
#         index = index % 1000000
#         for line in f:
#             data = json.loads(line)
#             if index == 0:
#                 return [data["title"], data["entities"], data["paperAbstract"]]
#             index -= 1

def getBestArticlesIndexes():
    print("getting best articles indexes...")
    indexes = []
    bestVector = np.memmap(BEST_INDEXES_VEC, dtype="int64", mode="r+", shape=(BEST_ARTICLE_NUM))
    for i in range(BEST_ARTICLE_NUM):
        indexes.append(bestVector[i])
    return indexes

def getGoodData():
    indexes = getBestArticlesIndexes()
    newIndexes = []
    textDatas = getTextDatasFromBestTxtVec()
    # print(textDatas)
    newData = []
    abstracts = []
    titles = []
    for i in range(BEST_ARTICLE_NUM):
        data = textDatas[i]
        if (len(data[2].split(" ")) > 7 and "Archive" not in data[1]):  # if the abstract exists and is not too small and not a jstor archive
            newIndexes.append(indexes[i])
            newData.append(data)
            abstracts.append(textDatas[i][2])
            titles.append(textDatas[i][0])
    indexes = newIndexes
    # textDatas = newData
    return indexes, titles, abstracts


if __name__ == '__main__':
    createBestArticlesIndexesVec()
    createBestTextDataDatabse(getBestArticlesIndexes())
    print(getGoodData())


