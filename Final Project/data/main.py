from rank_vector import RankVector
from edge_matrix import EdgeMatrix



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