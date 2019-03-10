import sql_map
import time
import os
import json

from constants import *


def create_researcher():
    return {
        "latest" : "0"
    }


def create_resercher_sql():
    index = 0
    researchers = sql_map.SqlMap("RESEARCHERS", RESEARCHERS_PATH)
    start_time = time.time()

    for filename in sorted(os.listdir(RAW_DATA_PATH), key=lambda x: int(x)):
        pathname = RAW_DATA_PATH + filename
        print(pathname)

        with open(pathname, 'rb') as f:
            for line in f:
                data = json.loads(line)

                # extract all the stuff about each researcher
                for author in data["authors"]:
                    try:
                        author_id = int(author["ids"][0])

                        # get current author data
                        author_obj = researchers[author_id]
                        if author_obj is None:
                            author_obj = create_researcher()
                        else:
                            author_obj = dict(author_obj)

                        # set new author features
                        if data["year"] != "":
                            author_obj["latest"] = max(int(author_obj["latest"]), int(data["year"]))

                        # update new author
                        researchers[author_id] = str(author_obj)

                        print(author_obj)
                    except Exception as e:
                        print(e)

                # move on
                index += 1
                if not index % 1000:
                    print(index, "\tFinished. Elapsed time: %.2f sec" % (time.time() - start_time))
                    researchers.save()
    researchers.save()
    researchers.close()

create_resercher_sql()