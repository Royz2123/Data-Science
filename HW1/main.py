import re

from selenium import webdriver

import util
import list_projects


def main():
    driver = webdriver.Firefox()
    projects = list_projects.get_project_links(driver)
    util.write_list(projects)
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

