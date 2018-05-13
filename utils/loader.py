# read the class names

def load_classes(namefile):
    fp = open(namefile, 'r')
    names = fp.read().split('\n')[:-1]
    return names