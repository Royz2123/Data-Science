import json
import matplotlib.pyplot as plt

DEFAULT_LINKS_FILE = "project_links.txt"
DEFAULT_OUTPUT_FILE = "json_data.json"

def write_list(lst, filename=DEFAULT_LINKS_FILE):
    with open(filename, 'w') as fp:
        for item in lst:
            fp.write("%s\n" % item)

def read_list(filename=DEFAULT_LINKS_FILE):
    with open (filename, 'r') as fp:
        return fp.read().split('\n')[:-1]

def write_dict(output):
    outputText = json.dumps(output, indent=4, )
    with open("json_data.json", 'w') as f:
        f.write(outputText)
        
def plot_in_R2(x, y, title="Y as a function of X"):
    plt.scatter(x, y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.show()