import os
import os.path as osp

ff = open('/data1/wlj/StateGridv5/验证集/杆塔/ImageSets/Main/val.txt','r')

xml_names = []
for line in ff:
    xml_names.append(line.split('\n')[0])


jpg_not_exists = []
for xml_name in xml_names:
    if not osp.exists(osp.join('/data1/wlj/StateGridv5/验证集/JPEGImages',xml_name+'.jpg')):
        jpg_not_exists.append(xml_name)
from pdb import set_trace
set_trace()
print(jpg_not_exists)