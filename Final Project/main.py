import sys
import time

from selenium import webdriver

import get_data
import list_projects
import util

MERCY_TIME = 1



def create_filename(topic, papers):
    return "data/paper_links_" + topic.replace(" ", "_") + "_" + str(papers) + ".txt"

def main():
    topic = "machine learning"
    papers = 200

    # Step 1: get links for projects
    if True:
        driver = webdriver.Firefox()
        projects = list_projects.get_project_links(driver, topic, num_of_projects=papers)
        util.write_list(projects, create_filename(topic, papers))
        driver.close()
    else:
        projects = util.read_list()

    """
    # Step 2: get data for every project
    output = dict()
    output["records"] = {"record": []}

    for project_index, project_link in enumerate(projects):
        output["records"]["record"].append(get_data.get_data_from_url(
            project_link,
            project_index + 1
        ))
        print("Crawled:\t%d/%d" % (project_index + 1, len(projects)))

        # Have mercy on KickStarter :)
        time.sleep(MERCY_TIME)

    # Write into JSON file
    util.write_dict(output)
    """


if __name__ == "__main__":
    main()

