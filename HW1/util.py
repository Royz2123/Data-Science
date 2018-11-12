DEFAULT_FILE = "project_links.txt"

def write_list(lst, filename=DEFAULT_FILE):
    with open(filename, 'w') as fp:
        for item in lst:
            fp.write("%s\n" % item)

def read_list(filename=DEFAULT_FILE):
    with open (filename, 'r') as fp:
        return fp.read().split('\n')[:-1]