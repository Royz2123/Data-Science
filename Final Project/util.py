import json
from constants import *
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LINKS_FILE = "paper_links.txt"
DEFAULT_OUTPUT_FILE = "json_data.json"


def plot_paper_ranks(papers, ticks=False):
    x = np.array(range(len(papers)))
    if ticks:
        y = np.array([paper[1] for paper in papers])
        my_xticks = [paper[0] for paper in papers]
        plt.xticks(x, my_xticks, rotation='vertical')
    else:
        y = papers
        plt.xlabel("Paper Indices")
    plt.plot(x, y)
    plt.title("Top " + str(len(papers)) + " papers, plotted against their rank")
    plt.ylabel("Rank")
    plt.show()


def write_list(lst, filename=DEFAULT_LINKS_FILE):
    with open(filename, 'w') as fp:
        for item in lst:
            fp.write("%s\n" % item)


def read_list(filename=DEFAULT_LINKS_FILE):
    with open(filename, 'r') as fp:
        return fp.read().split('\n')[:-1]


def write_dict(output):
    outputText = json.dumps(output, indent=4, )
    with open("json_data.json", 'w') as f:
        f.write(outputText)


def get_url_from_index(index):
    pathname = RAW_DATA_PATH + str(index // 1000000)
    with open(pathname, 'rb') as f:
        index = index % 1000000
        for line in f:
            data = json.loads(line)
            if index == 0:
                return len(data["inCitations"]), data["s2Url"]
            index -= 1


def get_paper_from_index(index):
    pathname = RAW_DATA_PATH + str(index // 1000000)
    with open(pathname, 'rb') as f:
        index = index % 1000000
        for line in f:
            data = json.loads(line)
            if index == 0:
                return data
            index -= 1


def dist(vec1, vec2):
    dist_val = 0
    for i in range(RECORDS):
        dist_val += abs(vec1[i] - vec2[i])
    return dist_val
