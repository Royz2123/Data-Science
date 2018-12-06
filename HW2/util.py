import json
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_LINKS_FILE = "project_links.txt"
DEFAULT_OUTPUT_FILE = "json_data.json"

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

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