import parser
from edge_matrix import EdgeMatrix
from constants import *

import numpy as np
import time
import json

import os

#edges = EdgeMatrix(True)
RECORDS = int(10 ** 7 * 3.918)
DEBUG_MODE = False

def get_url_from_index(index):
    pathname = RAW_DATA_PATH + str(index // 1000000)
    with open(pathname, 'rb') as f:
        index = index % 1000000
        for line in f:
            data = json.loads(line)
            if index == 0:
                return (len(data["inCitations"]), data["s2Url"])
            index -= 1


def dist(vec1, vec2):
    dist = 0
    for i in range(RECORDS):
        dist += abs(vec1[i] - vec2[i])
    return dist


def researcher_ranking(name):
    ranking = 0
    index = 0
    rank_vector = np.memmap("databases/RankVector0.01", dtype="float64", mode="r+", shape=(10**7 * 4))
    start_time = time.time()

    for filename in sorted(os.listdir(RAW_DATA_PATH), key=lambda x: int(x)):
        pathname = RAW_DATA_PATH + filename
        print(pathname)

        with open(pathname, 'rb') as f:
            for line in f:
                data = json.loads(line)
                if name in [author["name"] for author in data["authors"]]:
                    print(data["s2Url"])
                    ranking += rank_vector[index]
                index += 1
        print(ranking)

    print("Researcher name: ", name, "\tRanking: ", ranking)


def max_rank_paper():
    curr_vector = np.memmap("databases/RankVector0.01", dtype="float64", mode="r+", shape=(10**7 * 4))

    max_index = 0
    max_value = 0
    for i in range(RECORDS):
        curr_val = curr_vector[i]
        if curr_val > max_value:
            max_index = i
            max_value = curr_val

        if not i % 100000:
            print(i)

    print(curr_vector[max_index])
    print(get_url_from_index(max_index))



def rank_vector():
    start_time = time.time()
    edges = EdgeMatrix()

    open("databases/Vector1", 'w+').close()
    open("databases/Vector2", 'w+').close()
    curr_vector = np.memmap("databases/Vector1", dtype="float64", mode="r+", shape=(10**7 * 4))
    prev_vector = np.memmap("databases/Vector2", dtype="float64", mode="r+", shape=(10**7 * 4))
    out_citations = np.memmap(OUT_CITATIONS_PATH, dtype="int64", mode="r+", shape=(10 ** 8 * 4))

    for vec in (curr_vector, prev_vector):
        for i in range(RECORDS):
            vec[i] = 1.0 / RECORDS

    max_paper = None
    for iteration in range(MAX_ITERATIONS):
        for paper_index in range(RECORDS):
            new_rank = 0
            for index in edges[paper_index]:
                d = out_citations[index]
                if d == 0:
                    print("Conflict: Out citations are 0 for index ", index)
                else:
                    new_rank += prev_vector[index] / d
            curr_vector[paper_index] = (1 - LAMBDA) * new_rank + LAMBDA / RECORDS

            if max_paper is None or max_paper[1] < curr_vector[paper_index]:
                max_paper = (paper_index, curr_vector[paper_index], len(edges[paper_index]))

            """
            print()
            print("Before  ", prev_vector[paper_index])
            print("After   ", curr_vector[paper_index])
            print(curr_vector[paper_index] / prev_vector[paper_index])
            print()
            time.sleep(1)
            """

            if DEBUG_MODE and not paper_index % 100000:
                print(paper_index, "\tCalculated ranks. Elapsed time: %.2f sec" % (time.time() - start_time))
                print("Max Paper until now: ", max_paper)

        # check convergence
        distance = dist(curr_vector, prev_vector)
        print("Iteration no. ", iteration, "\tConvergence distance: ", distance, "\tTarget Distance: ", EPSILON)
        if distance < EPSILON:
            break

        # switch
        temp_vector = curr_vector
        curr_vector = prev_vector
        prev_vector = temp_vector


rank_vector()

"""
def test():
    edges = EdgeMatrix()

    for i in range(10):
        print(edges.get_cited(i))


def create_databases():
    #fetcher.fetch_raw_data()
    parser.store_raw_data()
    edges = EdgeMatrix()

    #curr_vector = Vector("RANKS", 1, RANK_VEC_1_PATH)
    #prev_vector = Vector("RANKS", 1, RANK_VEC_2_PATH)


MAIN PSEUDO CODE


def main():
    rank_vector = RankVector()
    prev_rank_vector = RankVector()
    edges = EdgeMatrix()

    curr_vector = rank_vector
    prev_vector = prev_rank_vector

    for iteration in range(10):
        for paper_index in range(prev_vector.length()):
            curr_vector[paper_index] = sum([prev_vector[index] for index in edges.get_cited(paper_index)])

        # add teleports
        # curr_vector[paper_index] +=

        # check convergence

        # switch
        temp_vector = curr_vector
        curr_vector = prev_vector
        prev_vector = temp_vector


"""