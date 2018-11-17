from collections import OrderedDict
import json
import re
import requests
import time

import util


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
    return item


def main():
    #reload(sys)
    #sys.setdefaultencoding('utf8')

    projects = util.read_list()
    output = OrderedDict()
    output["records"] = {"record": []}
    for i in range(30):
        output["records"]["record"].append(get_data_from_url(projects[i], i + 1))

    outputText = json.dumps(output, indent=4, )
    with open("json_data.json", 'w') as f:
        f.write(outputText)


if __name__ == '__main__':
    main()

