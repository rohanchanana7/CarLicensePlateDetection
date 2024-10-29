import os
import re
import cv2
import pandas as pd
import xml.etree.ElementTree as xet
from glob import glob

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(0)) if match else 0

def load_data(dataset_path):
    labels_dict = {
        'img_path': [], 'xmin': [], 'xmax': [], 'ymin': [], 'ymax': [], 'img_w': [], 'img_h': []
    }
    xml_files = sorted(glob(f'{dataset_path}/annotations/*.xml'), key=extract_number)

    for filename in xml_files:
        info = xet.parse(filename)
        root = info.getroot()

        member_object = root.find('object')
        labels_info = member_object.find('bndbox')
        xmin = int(labels_info.find('xmin').text)
        xmax = int(labels_info.find('xmax').text)
        ymin = int(labels_info.find('ymin').text)
        ymax = int(labels_info.find('ymax').text)

        img_name = root.find('filename').text
        img_path = os.path.join(dataset_path, 'images', img_name)

        labels_dict['img_path'].append(img_path)
        labels_dict['xmin'].append(xmin)
        labels_dict['xmax'].append(xmax)
        labels_dict['ymin'].append(ymin)
        labels_dict['ymax'].append(ymax)

        height, width, _ = cv2.imread(img_path).shape
        labels_dict['img_w'].append(width)
        labels_dict['img_h'].append(height)

    return pd.DataFrame(labels_dict)
