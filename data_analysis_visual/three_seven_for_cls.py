import argparse
import os
from re import I
import xml.etree.ElementTree as ET
import numpy as np
import cv2

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

inds = np.arange((len(xml_paths)))
np.random.shuffle(inds)

train_xml = [xml_paths[i] for i in inds[:int(0.7*len(xml_paths))]]
val_xml = [xml_paths[i] for i in inds[int(0.7*len(xml_paths)):]]

root = os.path.join(dirname, "ImageSets/Main/")

f = open(root+'train_cls.txt','w')
for xml in train_xml:
    f.write(os.path.basename(xml).split('.')[0]+'\n')

f.close()

f = open(root+'val_cls.txt','w')
for xml in val_xml:
    f.write(os.path.basename(xml).split('.')[0]+'\n')

f.close()

from pdb import set_trace
set_trace()