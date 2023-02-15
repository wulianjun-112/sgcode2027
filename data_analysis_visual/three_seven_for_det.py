import argparse
import os
from re import I
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')

    parser.add_argument(
        '--anno-dir',
        type=str,
        help='path of annotations file')
        

    args = parser.parse_args()
    return args

args = parse_args()

assert args.anno_dir.endswith('txt')

f = open(args.anno_dir,'r')
xml_paths = []
dirname = os.path.dirname(os.path.dirname(os.path.dirname(args.anno_dir)))
for line in f:
    xml_paths.append(os.path.join(dirname, "Annotations", line.split('\n')[0] + ".xml"))

f.close()

train_ids,val_ids = train_test_split(xml_paths,test_size=0.1,random_state=41)

from pdb import set_trace
set_trace()
root = os.path.join(dirname, "ImageSets/Main/")

f = open(root+'train_split.txt','w')
for xml in train_ids:
    f.write(os.path.basename(xml).split('.xml')[0]+'\n')

f.close()

f = open(root+'val_split.txt','w')
for xml in val_ids:
    f.write(os.path.basename(xml).split('.xml')[0]+'\n')

f.close()

