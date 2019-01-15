import time

#PAGE_URL = "https://www.kickstarter.com/discover/advanced?category_id=16&woe_id=0&sort=magic&seed=2569455&page="

DEFAULT_PAGE_URL = "https://www.semanticscholar.org/search?q=big%20data&sort=relevance&pdf=true&page="
QUERY = "big data"

CELL_URL_STYLE = "icon-button.paper-link"

DEFAULT_NUM = 300

SLEEP_TIME = 1

def create_query(topic, page_num):
    return (
        "https://www.semanticscholar.org/search?q="
        + topic
        + "&sort=relevance&pdf=true&page="
        + str(page_num)
    )

"""
Get all links from single webpage of results
"""
def get_links_from_page(driver, link_style=CELL_URL_STYLE):
    return [project.get_attribute('href') for project in driver.find_elements_by_class_name(link_style)]


"""
Get num_of_projects projects
"""
def get_project_links(driver, topic, num_of_projects=DEFAULT_NUM):
    project_links = []
    page_num = 1
    while len(project_links) < num_of_projects:
        driver.get(create_query(topic, page_num))

        # Have mercy on the website
        time.sleep(SLEEP_TIME)

        # Updates
        project_links += get_links_from_page(driver)
        page_num += 1

        # Output progress
        print("Progress:\t" + str(len(project_links)) + "/" + str(num_of_projects))
    return project_links[:num_of_projects]

