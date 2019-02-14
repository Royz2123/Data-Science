import fetcher
import parser
from rank_vector import Vector
from sql_edge_matrix import EdgeMatrix
from constants import *


#edges = EdgeMatrix(True)


def test():
    edges = EdgeMatrix(False)

    for i in range(1000, 20000):
        print(edges.get_cited(i))

    edges.close()


test()

def create_databases():
    fetcher.fetch_raw_data()
    parser.store_raw_data()
    edges = EdgeMatrix()

    curr_vector = Vector("RANKS", 1, RANK_VEC_1_PATH)
    prev_vector = Vector("RANKS", 1, RANK_VEC_2_PATH)


def main():
    edges = EdgeMatrix(False)

    curr_vector = Vector("RANKS", 1, RANK_VEC_1_PATH)
    prev_vector = Vector("RANKS", 1, RANK_VEC_2_PATH)

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