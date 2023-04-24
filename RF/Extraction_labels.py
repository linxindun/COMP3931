from xml.etree import ElementTree

def Extraction(path):
    tree = ElementTree.parse(path)
    root = tree.getroot()
    labels = []

    for obj in root.iter('object'):
        label = obj.find('name').text
        labels.append(label)

    return labels
