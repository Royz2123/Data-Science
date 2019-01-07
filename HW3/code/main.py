
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk import pos_tag
import matplotlib.pyplot as plt

nltk.download('averaged_perceptron_tagger')

STOPWORDS = stopwords.words("english")

def read_file(filename):
    with open(filename, "r") as fileobj:
        return fileobj.read()


def pretty_print(lst):
    for index, elem in enumerate(lst):
        print(index + 1, "\t", str(elem))

def plot_log_log(x, y):
    plt.loglog(x, y)
    plt.title("Log-Log plot of word frequency as a function of rank", fontsize=12)
    plt.xlabel("Log Rank")
    plt.ylabel("Log Frequency")
    plt.show()



# Question 2
def question2():
    txt = read_file("sherlock_holmes.txt")
    txt = nltk.word_tokenize(txt)
    txt = list(filter(lambda x: x.isalpha(), txt))
    occ = [(word, txt.count(word)) for word in list(set(txt))]
    occ = sorted(occ, key=lambda x: x[1], reverse=True)
    top = occ[:20]
    pretty_print([val[0] for val in top])
    plot_log_log(list(range(1, len(occ) + 1)), [val[1] for val in occ])


# Question 3
def question3():
    txt = read_file("sherlock_holmes.txt")
    txt = nltk.word_tokenize(txt)
    txt = list(filter(lambda x: x.isalpha() and x.lower() not in STOPWORDS, txt))
    occ = [(word, txt.count(word)) for word in list(set(txt))]
    occ = sorted(occ, key=lambda x: x[1], reverse=True)
    top = occ[:20]
    pretty_print([val[0] for val in top])
    plot_log_log(list(range(1, len(occ) + 1)), [val[1] for val in occ])

# Question 4
def question4():
    txt = read_file("sherlock_holmes.txt")
    txt = nltk.word_tokenize(txt)
    txt = list(filter(lambda x: x.isalpha() and x.lower() not in STOPWORDS, txt))
    stemmer = PorterStemmer()
    txt = [stemmer.stem(val) for val in txt]
    occ = [(word, txt.count(word)) for word in list(set(txt))]
    occ = sorted(occ, key=lambda x: x[1], reverse=True)
    top = occ[:20]
    pretty_print([val[0] for val in top])
    plot_log_log(list(range(1, len(occ) + 1)), [val[1] for val in occ])


def question5():
    txt = read_file("sherlock_holmes.txt")
    txt = nltk.word_tokenize(txt)
    tagged = pos_tag(txt)
    nouns = []
    i = 0
    while i < len(tagged):
        noun = ""
        while  i < len(tagged) and tagged[i][1] in ["JJ", "JJR", "JJS"]:
            word, type = tagged[i]
            noun += " " + word
            i = i+1
        if i >= len(tagged) or tagged[i][1] not in ["NN", "NNP", "NNS", "NNPS"]:
            i += 1
            continue
        while i < len(tagged) and tagged[i][1] in ["NN", "NNP", "NNS", "NNPS"]:
            word, type = tagged[i]
            noun += " " + word
            i = i+1
        nouns.append(noun[1:])
    occ = [(noun, nouns.count(noun)) for noun in list(set(nouns))]
    occ = sorted(occ, key=lambda x: x[1], reverse=True)
    top = occ[:20]
    pretty_print([val[0] for val in top])
    plot_log_log(list(range(1, len(occ) + 1)), [val[1] for val in occ])


# question5()
question3()



