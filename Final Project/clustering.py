from sklearn.externals import joblib
from wordcloud import WordCloud

from top_rank_articles import getGoodData
import nltk as nltk
import numpy as np
import re
import pandas as pd
import os

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt

from constants import *

import edge_matrix
import sql_map

STOPWORDS = stopwords.words("english")
stemmer = SnowballStemmer("english")


def isGoodWord(token):
    return (
        re.search('[a-zA-Z]', token)
        and token != "italic"
        and len(token) > 1
        and not (len(token) == 2 and '.' in token)
        and token != 'lb'
        and '/' not in token
        and '\'' not in token
    )


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if isGoodWord(token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if isGoodWord(token):
            filtered_tokens.append(token)
    return filtered_tokens


def cluster_best_articles(type="KMeans", clusterNum=5):
    print(clusterNum)
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
    tfidf_vectorizer = TfidfVectorizer(max_df=0.3, max_features=200000, # tune able hyper parameters
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

        # Save Clusters
        clustering_title = str(clusterNum) + "_Cluster" + str(BEST_ARTICLE_NUM) + "_Articles"

        try:
            os.mkdir("./clusters/" + clustering_title)
        except:
            pass

        for i in range(clusterNum):
            getWordCloudForCentroid(
                km,
                i,
                terms,
                vocab_frame,
                clustering_title
            )
    else:
        print("Clustering type not supported yet")


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def getWordCloudForCentroid(km, i, terms, vocab_frame, clustering_title="New_Cluster"):
    print("getting wordcloud for cluster " + str(i))
    clusterNum = i
    centroids = pd.DataFrame(km.cluster_centers_)
    a = tuple(tuple(centroids.loc[i, :].reset_index().values))
    unstemmer = vocab_frame.to_dict('dict')['words']
    freq = {}
    for (i, f) in a:
        i = int(i)
        shingle = ""
        for w in terms[i].split(' '):
            # shingle += vocab_frame.loc[w].values.tolist()[0][0] + " "
            if (w in unstemmer):
                shingle += unstemmer[w] + " "
        if not np.isnan(f) and not f == 0:
            freq[shingle] = f

    wordcloud = WordCloud(width=800, height=400, background_color='white')
    kMeansWordCloud = wordcloud.generate_from_frequencies(freq)
    plt.figure(figsize=(20, 10))
    title = "Cluster " + str(clusterNum)
    plt.title(title, fontdict={'fontsize': 32, 'fontweight': 'medium'})
    plt.imshow(kMeansWordCloud)
    plt.axis("off")
    plt.savefig("./clusters/" + clustering_title + "/" + title)
    # plt.show()


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
    tfidf_vectorizer = TfidfVectorizer(max_df=0.3, max_features=200000, # tune able hyper parameters
                                       stop_words='english',
                                       use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(abstracts)

    vectorizedAbstract = tfidf_vectorizer.transform([abstract])
    km = joblib.load('doc_cluster.pkl') #if we already did and saved the kmeans clustering
    clusters = km.labels_.tolist()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    # #error here, something wih dimensions
    # dist = [1 - cosine_sim(vectorizedAbstract.toarray(), c) for c in order_centroids]
    # print(dist)
    i = km.predict(vectorizedAbstract)[0]

    terms = tfidf_vectorizer.get_feature_names()
    papers = {'title': titles, 'abstract': abstracts, 'cluster': clusters}
    frame = pd.DataFrame(papers, index=[clusters], columns=['title', 'cluster'])

    print("This paper was classified to the following cluster:")

    print("Cluster %d: " % i, end='')

    clusterTitle = []
    for ind in order_centroids[i, :20]:  # replace 6 with n words per cluster
        clusterTitle.append(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0])
    t = ", ".join(clusterTitle)
    print(t)
    print()

    # print("Cluster %d titles:" % i, end='')
    # for title in frame.loc[i]['title'].values.tolist():
    #     print(' %s,' % title, end='')
    print()
    print()

    getWordCloudForCentroid(km, i, terms, vocab_frame)


#if __name__ == '__main__':
#    cluster_best_articles("KMeans", 20)
