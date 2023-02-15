import os.path as osp

import os
import xml.etree.ElementTree as ET
import numpy as np

jpgs = os.listdir('/data/wulianjun/StateGridv5/验证集/杆塔/JPEGImages/')
jpgs = [osp.basename(x).split('.jpg')[0] for x in jpgs]
xmls = os.listdir('/data/wulianjun/StateGridv5/验证集/杆塔/Annotations/')
xmls = [osp.basename(x).split('.xml')[0] for x in xmls]

no_jpg_xml = []
for x in xmls:
    if x not in jpgs:
        no_jpg_xml.append(x)
from pdb import set_trace
set_trace()