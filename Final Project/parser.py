# each JSON is small, there's no need in iterative processing
import json
import time


def read_raw_data(filename):
    with open(filename, 'rb') as f:
        for line in f:
            data = json.loads(line)
            article = {
                "name" : ""
            }
            print(data.keys())
            print(data)
            time.sleep(1)
            # data[u'name'], data[u'engine_speed'], data[u'timestamp'] now
            # contain correspoding values

read_raw_data("real_data/2")