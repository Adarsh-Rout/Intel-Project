import xml.etree.ElementTree as ET
import os

def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    annotations = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        annotations.append((name, xmin, ymin, xmax, ymax))
    
    return annotations

# xml_file = r'idd-detection\IDD_Detection\Annotations\frontFar\BLR-2018-03-22_17-39-26_2_frontFar\000006_r.xml'
# annotations = parse_annotation(xml_file)
# print(annotations)