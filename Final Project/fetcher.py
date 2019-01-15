import urllib.request
import time

DEFAULT = "https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/"
DESTINATION = "raw_data/"
TOTAL_PAGES = 40


def fetch_raw_data():
	corpus_index = 0
	while corpus_index < TOTAL_PAGES:
		try:
			print("Extracting Corpus " + str(corpus_index) + "/" + "40")
			path = DEFAULT + "corpus-2018-05-03/s2-corpus-" + str(corpus_index).zfill(2) + ".gz"
			urllib.request.urlretrieve(path, "raw_data/" + str(corpus_index) + ".gz")
		except Exception as e:
			# Have mercy on the website, probably blocking us
			print(e)
			time.sleep(100)
			corpus_index -= 1
		finally:
			corpus_index += 1
