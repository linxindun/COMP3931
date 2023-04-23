from xml.etree import ElementTree

def Extraction(path):
    tree = ElementTree.parse(path)
    root = tree.getroot()
    location = []
    labels = []

    for obj in root.iter('object'):
        label = obj.find('name').text
        labels.append(label)

        box = obj.find('bndbox')
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        location.append([xmin, ymin, xmax, ymax])

    return location, labels
