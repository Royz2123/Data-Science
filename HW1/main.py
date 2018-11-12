import re

from selenium import webdriver
import pickle
from selenium.webdriver.common.keys import Keys


BASE_URL = "https://www.kickstarter.com/discover/categories/technology"
PAGE_URL = "https://www.kickstarter.com/discover/advanced?category_id=16&woe_id=0&sort=magic&seed=2569455&page="
CELL_URL_STYLE = "soft-black.mb3"

DEFAULT_NUM = 300
DEFAULT_FILE = "project_links.txt"


def write_list(lst, filename=DEFAULT_FILE):
    with open(filename, 'w') as fp:
        for item in lst:
            fp.write("%s\n" % item)

def read_list(filename=DEFAULT_FILE):
    with open (filename, 'r') as fp:
        return fp.read().split('\n')[:-1]

def get_links_from_page(driver, link_style=CELL_URL_STYLE):
    return [project.get_attribute('href') for project in driver.find_elements_by_class_name(link_style)]


def get_project_links(driver, num_of_projects=DEFAULT_NUM):
    project_links = []
    page_num = 1
    while len(project_links) < num_of_projects:
        driver.get(PAGE_URL + str(page_num))
        project_links += get_links_from_page(driver)
        page_num += 1
    return project_links[:num_of_projects]


def main():
    driver = webdriver.Firefox()
    projects = get_project_links(driver)
    write_list(projects)
    driver.close()


if __name__ == "__main__":
    main()




"""

def main():
    contents = get_results_page(1)
    print("Real Racer" in str(contents))
    #contents = "<a href ='https://example.com' class='block img-placeholder w100p'></a>"
    # parse
    print(CELL_URL_STYLE in str(contents))
    html = BeautifulSoup(contents, 'html.parser')

    print(html.findAll(lambda tag: tag.name == 'div'))

    # print(get_cell_list(html))


    #print(len(links))

    #print(links)

    #print(list(links[0].children)[0])

<a href="https://www.kickstarter.com/projects/meadow/meadow-full-stack-net-standard-iot-platform?ref=discovery" class="block img-placeholder w100p"><img class="border-grey-400 border-bottom w100p absolute t0" alt="Project image" src="https://ksr-ugc.imgix.net/assets/022/746/799/58401c452710a07066ffc62cac3cfa05_original.png?ixlib=rb-1.1.0&amp;crop=faces&amp;w=560&amp;h=315&amp;fit=crop&amp;v=1540878238&amp;auto=format&amp;frame=1&amp;q=92&amp;s=66481006d073b6bc8e9d3227690f2396"></a>

 interesting_cells = []

    links = soup.find_all(class_=cell_style)

    # hrefs = [link.get('href') for link in soup.find_all('a')]
    # print(hrefs)

    print(links)

    for link in links:
        if

        if link.has_attr("class"):# and link.has_attr("href"):
            print(" ".join(link["class"]))
            if "soft" in " ".join(link["class"]):
                print(link)
            if link["class"][0] == "clamp-5":
                print(link)
"""

