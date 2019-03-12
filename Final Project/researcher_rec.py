import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

import sql_map
import time
import os
import json

from constants import *

RESEARCHERS = 291454


def get_top_keywords(abstract):
    r = Rake()

    # extracting the words by passing the text
    r.extract_keywords_from_text(abstract)

    # getting the dictionary with key words as keys and their scores as values
    key_words_dict_scores = r.get_word_degrees()

    # sort by scores
    top_keywords = sorted(key_words_dict_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    # assigning the key words to the new column for the corresponding movie
    return " ".join([word[0] for word in top_keywords])


def create_vector_from_paper(data):
    paper_keywords = get_top_keywords(data["title"] + data["paperAbstract"]) + " " + " ".join(data["entities"])
    paper_authors = " ".join(auth["name"].replace(" ", "") for auth in data["authors"])
    return (paper_authors + " " + paper_keywords + " ").lower()


def create_researcher_vector(researcher_id, researcher_name):
    researcher_vector = ""
    for filename in sorted(os.listdir(RAW_DATA_PATH), key=lambda x: int(x)):
        pathname = RAW_DATA_PATH + filename
        print(pathname)

        with open(pathname, 'rb') as f:
            for line in f:
                data = json.loads(line)
                ids = [int(auth["ids"][0]) for auth in data["authors"] if auth["ids"] != []]
                names = [auth["name"] for auth in data["authors"]]

                if researcher_id in ids or researcher_name in names:
                    researcher_vector += create_vector_from_paper(data)
                    print("found article: ", data["title"])
                    break
    return researcher_vector


def find_similar_researchers(researcher_id=None, researcher_name=None):
    curr_vector = create_researcher_vector(researcher_id, researcher_name)
    researchers = sql_map.SqlMap("RESEARCHERS", RESEARCHERS_PATH)
    researchers_ids = np.memmap(RESEARCHER_IDS_VEC, dtype="int64", mode="r+", shape=10**8)

    vectors = [curr_vector]
    for i in range(20000):
        vectors.append(researchers[str(researchers_ids[i])].split("||")[1])

    # instantiating and generating the count matrix
    count = CountVectorizer()
    count_matrix = count.fit_transform(list(vectors))

    # generating the cosine similarity matrix
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    score_series = pd.Series(cosine_sim[0]).sort_values(ascending=False)
    top_10_indexes = list(score_series.iloc[1:11].index)
    top_10_researcher_ids = [researchers_ids[index - 1] for index in top_10_indexes]

    print("Most similar researches list:\n")
    print("Subject: ")
    for index, auth_id in enumerate(top_10_researcher_ids):
        print(index + 1, ":\t", researchers[str(auth_id)].split("||")[0].split("Name:", 2)[1])


def create_researcher_sql():
    # create databases
    researchers = sql_map.SqlMap("RESEARCHERS", RESEARCHERS_PATH)
    open(RESEARCHER_IDS_VEC, "w+").close()
    researchers_ids = np.memmap(RESEARCHER_IDS_VEC, dtype="int64", mode="r+", shape=10**8)

    # initialize
    index = 0
    researchers_index = 0
    rank_vector = np.memmap("databases/RankVector06", dtype="float64", mode="r+", shape=(10**7 * 4))
    start_time = time.time()

    for filename in sorted(os.listdir(RAW_DATA_PATH), key=lambda x: int(x)):
        pathname = RAW_DATA_PATH + filename
        print(pathname)

        with open(pathname, 'rb') as f:
            for line in f:
                if rank_vector[index] >= 10**-7:
                    data = json.loads(line)

                    # look only at relevant researchers (Last 20 years)
                    if (
                        "year" not in data.keys()
                        or data["year"] == ""
                        or int(data["year"]) < 1990
                    ):
                        continue

                    # paper info
                    new_paper_vector = create_vector_from_paper(data)

                    # extract all the stuff about each researcher
                    for author in data["authors"]:
                        if len(author["ids"]):
                            author_id = int(author["ids"][0])

                            # get current author data
                            author_vector = researchers[author_id]
                            if author_vector is None:
                                author_vector = "Name: " + author["name"] + " || "

                            # update new author
                            researchers[author_id] = author_vector + new_paper_vector
                            researchers_ids[researchers_index] = int(author_id)
                            researchers_index += 1

                # move on
                index += 1
                if not index % 100000:
                    print(index, "\tFinished. Elapsed time: %.2f sec" % (time.time() - start_time))
                    researchers.save()
    researchers.save()
    researchers.close()
    print("TOTAL RESEARCHERS: ", researchers_index)


#if __name__ == "__main__":
#    create_researcher_sql()
#    find_similar_researchers(32288193)
