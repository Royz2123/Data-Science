PAGE_URL = "https://www.kickstarter.com/discover/advanced?category_id=16&woe_id=0&sort=magic&seed=2569455&page="
CELL_URL_STYLE = "soft-black.mb3"

DEFAULT_NUM = 300

"""
Get all links from single webpage of results
"""
def get_links_from_page(driver, link_style=CELL_URL_STYLE):
    return [project.get_attribute('href') for project in driver.find_elements_by_class_name(link_style)]


"""
Get num_of_projects projects
"""
def get_project_links(driver, num_of_projects=DEFAULT_NUM):
    project_links = []
    page_num = 1
    while len(project_links) < num_of_projects:
        driver.get(PAGE_URL + str(page_num))
        project_links += get_links_from_page(driver)
        page_num += 1
    return project_links[:num_of_projects]