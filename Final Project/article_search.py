import numpy as np
import os
from fuzzywuzzy import fuzz
import json
import itertools

import nltk
from nltk.corpus import stopwords

from constants import *
import util

STOPWORDS = stopwords.words("english")
TOP_RESULTS = 10


def max_string_dist2(lst1, lst2):
    ratios = [(fuzz.ratio(w1, w2), w1, w2) for w1, w2 in list(itertools.product(lst1, lst2))]
    #print(ratios)
    #print(max(ratios, key=lambda x: x[0]))
    #if max(ratios, key=lambda x: x[0])[0] > 99:
    #    print(max(ratios, key=lambda x: x[0]))
    return max(ratios, key=lambda x: x[0])


def max_string_dist(lst1, lst2):
    return [len(set(lst1).intersection(lst2)) * 100]


def prep_strings(txt):
    txt = nltk.word_tokenize(txt)
    txt = list(filter(lambda x: x.isalpha() and x.lower() not in STOPWORDS, txt))
    return txt


def score_rank_merge(score, ranking):
    return score * np.log(1 + np.log((10**9 * ranking)))


def calc_score(keywords, data, ranking):
    #if ranking < 1e-9:
    #    return 0
    score = 0
    authors = [author["name"] for author in data["authors"]]

    title = prep_strings(data["title"])
    abstract = prep_strings(data["paperAbstract"]) + data["entities"]

    # check if same author, give big bonus if so
    if len(authors) and max_string_dist(authors, keywords)[0] > 90:
        score += 100
    if data["journalName"] != "" and max_string_dist([data["journalName"]], keywords)[0] > 90:
        score += 10
    if len(title):
        score += sum([max_string_dist([word], title)[0] > 90 for word in keywords]) ** 2
    if len(abstract):
        score += 0.5 * sum([max_string_dist([word], abstract)[0] > 90 for word in keywords]) ** 2
    return score_rank_merge(score, ranking)


def results_list(query, results):
    print("Results for '", query, "':")
    for i, result in enumerate(results):
        data = util.get_paper_from_index(result[0])
        print(
            "%d. %s (Match Score: %.5f)(%d):\n\t\tAuthors: %s\n\t\tAbstract:\t%s%s\n\t\tLink:\t%s\n" % (
                i+1,
                data["title"],
                result[1],
                result[0],
                ", ".join([author["name"] for author in data["authors"]]),
                data["paperAbstract"][:150],
                "..." * (len(data["paperAbstract"]) > 150),
                data["s2Url"]
            )
        )


def article_search(query):
    ranking = 0
    index = 0
    rank_vector = np.memmap("databases/RankVector06", dtype="float64", mode="r+", shape=(10 ** 7 * 4))
    keywords = prep_strings(query)
    print("Query: ", query, " Keywords: ", keywords)
    top_results = []

    for filename in sorted(os.listdir(RAW_DATA_PATH), key=lambda x: int(x)):
        pathname = RAW_DATA_PATH + filename
        with open(pathname, 'rb') as f:
            for line in f:
                data = json.loads(line)
                top_results.append((index, calc_score(keywords, data, rank_vector[index])))

                index += 1
                if not index % 10000:
                    top_results = sorted(top_results, key=lambda x: x[1], reverse=True)[:TOP_RESULTS]
                    print(index)
                    results_list(query, top_results)


article_search("Big Data")
