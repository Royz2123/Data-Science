from collections import OrderedDict
import re
import requests
import time


"""
Function that extracts the necessary data from url
@:param     url - the url we are extracting
            i - the index of the current page
@:return    returns a JSON object with the data
"""
def get_data_from_url(url, i):
    req = requests.get(url)
    htmltext = req.text
    s = re.search(r'converted_pledged_amount&quot;:(\d+)'
                  , htmltext)
    pledgeSum = s.group(1)
    s = re.search(r'&quot;goal&quot;:(\d+)'
                  , htmltext)
    goal = s.group(1)
    s = re.search(r'backers_count&quot;:(\d+)'
                  , htmltext)
    numbackers = s.group(1)

    s = re.search(r'</script><title>(.+) &'
                  , htmltext)
    if (s != None) :
        pass
    else:
        s = re.search(r'</script>\n<title>\n(.+) &'
                      , htmltext)
    title, creator = s.group(1).split(" by ")

    s = re.search(r'deadline&quot;:(\d+)'
                  , htmltext)
    daysLeft = -int((time.time() - int(s.group(1))) / 86400)


    item = OrderedDict()
    item["id"] = i
    item["url"] = url
    item["Creator"] = creator
    item["Title"] = title
    item["Text"] = htmltext
    item["DollarsPledged"] = pledgeSum
    item["DollarsGoal"] = goal
    item["NumBackers"] = numbackers
    item["DaysToGo"] = daysLeft
    item["AllOrNothing"] = True

    # call bonus on item
    bonus(item, htmltext)

    return item


"""
Function that extracts the bonus data from url
@:param     item - the html object we are inserting into
            htmltext - html of thepage we are extracting from
@:return    None
"""
def bonus(item, htmltext):
    s = re.findall(r'&quot;minimum&quot;:(\d+)', htmltext)

    price = s[:-1]
    s = re.findall(r';backers_count&quot;:(\d+)'
                   , htmltext)
    bacckerCount = s
    s = re.findall(r'&quot;limit&quot;:(\d+)'
                   , htmltext)
    maxBackers = s
    text = re.findall(r'--expanded">\n<p>(.+)' +
                      r'\b', htmltext)

    item["Rewards"] = OrderedDict()
    item["Rewards"] = []
    for i in range(1, len(price)):
        item["Rewards"].append({})
        text[i - 1] = text[i - 1].strip("</p")
        text[i - 1] = text[i - 1].replace('&#39;', '\'')
        text[i - 1] = text[i - 1].replace('&amp;', '&')
        item["Rewards"][i - 1]["Text"] = text[i - 1]
        item["Rewards"][i - 1]["Price"] = price[i]
        item["Rewards"][i - 1]["NumBackers"] = bacckerCount[i]
        if (len(maxBackers) > i):
            item["Rewards"][i - 1]["TotalPossible"] = maxBackers[i - 1]
        else:
            item["Rewards"][i - 1]["TotalPossible"] = "inf"

