from sklearn.externals import joblib

from sources.best_indexes_data_vectors import getGoodData
import nltk as nltk
import numpy as np
import time
import re
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from scipy.cluster.hierarchy import ward, dendrogram
from scipy.cluster.hierarchy import cut_tree
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt

from constants import *

import edge_matrix
import sql_map

STOPWORDS = stopwords.words("english")
stemmer = SnowballStemmer("english")


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token) and token != "italic" and len(token) > 1:
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token) and token != "italic" and len(token) > 1:
            filtered_tokens.append(token)
    return filtered_tokens


def cluster_best_articles(type="KMeans", clusterNum=5):
    indexes, titles, abstracts = getGoodData()

    totalvocab_stemmed = []
    totalvocab_tokenized = []
    for i in abstracts:
        allwords_stemmed = tokenize_and_stem(i)  # for each item in 'synopses', tokenize/stem
        totalvocab_stemmed.extend(allwords_stemmed)  # extend the 'totalvocab_stemmed' list

        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)

    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_stemmed)

    print("starting clustering...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.7, max_features=200000, # tune able hyper parameters
                                       stop_words='english',
                                       use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(abstracts)
    terms = tfidf_vectorizer.get_feature_names()
    if type == "KMeans":
        num_clusters = clusterNum

        km = KMeans(n_clusters=num_clusters)

        km.fit(tfidf_matrix)

        clusters = km.labels_.tolist()

        from sklearn.externals import joblib

        joblib.dump(km, 'doc_cluster.pkl')

        # km = joblib.load('doc_cluster.pkl') #if we already did and saved the kmeans clustering
        # clusters = km.labels_.tolist()

        print("clustering ended")
        papers = {'title': titles, 'abstract': abstracts, 'cluster': clusters}

        frame = pd.DataFrame(papers, index=[clusters], columns=['title', 'cluster'])
        print("Top terms per cluster:")
        print()
        # sort cluster centers by proximity to centroid
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        cluster_names = {}

        for i in range(num_clusters):
            print("Cluster %d words:" % i, end='')

            clusterTitle = []
            for ind in order_centroids[i, :20]:  # replace 6 with n words per cluster
                clusterTitle.append(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0])
            t = ", ".join(clusterTitle)
            cluster_names[i] = t[:3]  # save 3 most common words for later visualization(not implemented yet)
            print(t)
            print()  # add whitespace

            print("Cluster %d titles:" % i, end='')
            for title in frame.loc[i]['title'].values.tolist():
                print(' %s,' % title, end='')
            print()  # add whitespace
            print()  # add whitespace

        print()
        print()
    else:
        dist = 1 - cosine_similarity(tfidf_matrix)
        linkage_matrix = ward(dist)  # define the linkage_matrix using ward clustering pre-computed distances


        # from scipy.cluster.hierarchy import fcluster # probably can be used to know that are the colored clusters
        # tree = cut_tree(linkage_matrix) #maybe can be used to know that are the colored clusters


        fig, ax = plt.subplots(figsize=(30, 50)) # set size
        # ax = augmented_dendrogram(linkage_matrix, orientation="right", labels=titles)
        dendrogram(linkage_matrix,
                       # color_threshold=1,
                       p=100,
                       truncate_mode='lastp',
                       )
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)

        plt.tight_layout()
        plt.show()
        # plt.savefig('ward_clusters.png', dpi=200)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
def classifyPaper(abstract):
    indexes, titles, abstracts = getGoodData()

    totalvocab_stemmed = []
    totalvocab_tokenized = []
    for i in abstracts:
        allwords_stemmed = tokenize_and_stem(i)  # for each item in 'synopses', tokenize/stem
        totalvocab_stemmed.extend(allwords_stemmed)  # extend the 'totalvocab_stemmed' list

        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)

    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_stemmed)

    print("starting clustering...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.7, max_features=200000, # tune able hyper parameters
                                       stop_words='english',
                                       use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(abstracts)

    vectorizedAbstract = tfidf_vectorizer.transform([abstract])
    km = joblib.load('doc_cluster.pkl') #if we already did and saved the kmeans clustering
    clusters = km.labels_.tolist()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    #error here, something wih dimensions
    dist = [(1-cosine_similarity(vectorizedAbstract, order_centroids[i]) for i in range(len(order_centroids)))]
    print(dist)
    i = dist.index(min(dist))

    terms = tfidf_vectorizer.get_feature_names()
    papers = {'title': titles, 'abstract': abstracts, 'cluster': clusters}
    frame = pd.DataFrame(papers, index=[clusters], columns=['title', 'cluster'])

    print("This paper was classified to the following cluster:")

    print("Cluster %d words:" % i, end='')
    clusterTitle = []
    for ind in order_centroids[i, :20]:  # replace 6 with n words per cluster
        clusterTitle.append(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0])
    t = ", ".join(clusterTitle)
    print(t)
    print()  # add whitespace

    print("Cluster %d titles:" % i, end='')
    for title in frame.loc[i]['title'].values.tolist():
        print(' %s,' % title, end='')
    print()  # add whitespace
    print()  # add whitespace


# cluster_best_articles("KMeans", 10)
classifyPaper("Three biochemical methods depending on measurement of absorption in the ultraviolet range were successfuly adapted to histochemical techniques to be used in conjunction with ultraviolet microscopy. They included cetyl pyridinium chloride (max. = ca. 260 nm) as a stain for acid mucopolysaccharides; picryl sulfonic acid (max. = ca. 340 nm) as a stain for proteins, particularly amino groups; and fluorescein mercuric acetate (max. = 293 nm) as a stain for protein bound sulfhydryl and disulfide groups. The general possibilities of this approach was discussed")
# cluster_best_articles("KMeans", 10)

import os  # for os.path.basename

import matplotlib as mpl

# from sklearn.manifold import MDS
#
# MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
# mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
#
# pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
#
# xs, ys = pos[:, 0], pos[:, 1]
# print()
# print()
#
# #set up colors per clusters using a dict
# cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
#
# #set up cluster names using a dict
#
# # some ipython magic to show the matplotlib plots inline
#
# # create data frame that has the result of the MDS plus the cluster numbers and titles
# df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))
#
# # group by cluster
# groups = df.groupby('label')
#
# # set up plot
# fig, ax = plt.subplots(figsize=(17, 9))  # set size
# ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
#
# # iterate through groups to layer the plot
# # note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
# for name, group in groups:
#     ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
#             label=cluster_names[name], color=cluster_colors[name],
#             mec='none')
#     ax.set_aspect('auto')
#     ax.tick_params(
#         axis='x',  # changes apply to the x-axis
#         which='both',  # both major and minor ticks are affected
#         bottom='off',  # ticks along the bottom edge are off
#         top='off',  # ticks along the top edge are off
#         labelbottom='off')
#     ax.tick_params(
#         axis='y',  # changes apply to the y-axis
#         which='both',  # both major and minor ticks are affected
#         left='off',  # ticks along the bottom edge are off
#         top='off',  # ticks along the top edge are off
#         labelleft='off')
#
# ax.legend(numpoints=1)  # show legend with only 1 point
#
# # add label in x,y position with the label as the film title
# for i in range(len(df)):
#     ax.text(df.loc[i]['x'], df.loc[i]['y'], df.loc[i]['title'], size=8)
#
# plt.show()  # show the plot

# uncomment the below to save the plot if need be
# plt.savefig('clusters_small_noaxes.png', dpi=200)

#
#
#
#