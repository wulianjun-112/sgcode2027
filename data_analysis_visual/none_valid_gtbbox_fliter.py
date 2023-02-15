#coding=utf-8
import os
import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm

non_valid_xml = []

def read_xml(xml_path,dataset_class,relabel=False,save_path=None):
    DomTree = ET.parse(xml_path)
    file_name = osp.basename(xml_path)
    root = DomTree.getroot()
    size = root.find('size')
    if size is not None:
        width = int(size.find('width').text)
        height = int(size.find('height').text)
    if not relabel:
        for objects in root.findall('object'):
            label = objects.find('name').text
            if label in dataset_class:
                bnd_box = objects.find('bndbox')
                bbox = [
                            int(float(bnd_box.find('xmin').text)),
                            int(float(bnd_box.find('ymin').text)),
                            int(float(bnd_box.find('xmax').text)),
                            int(float(bnd_box.find('ymax').text))
                        ]
                # from pdb import set_trace
                # set_trace()
                if bbox[0]>=bbox[2] or bbox[1]>=bbox[3] or bbox[0]<0 or bbox[1]<0 or bbox[2]>width or bbox[3]>height:
                    non_valid_xml.append(xml_path)
                    return xml_path
    else:
        return_xml=False
        for objects in root.findall('object'):
            label = objects.find('name').text
            flag = np.array([c in label for c in dataset_class])
            if flag.sum()>1:
                from pdb import set_trace
                set_trace()
            if flag.any():
                objects.find('name').text = np.array(dataset_class)[flag].item()
                return_xml = True
        if return_xml:
            assert save_path is not None
            new_tree = ET.ElementTree(root)
            new_tree.write(osp.join(save_path,file_name),encoding='utf-8')
            return xml_path

# dataset_train_classes=[['推土机','塔吊','吊车','挖掘机']]
dataset_train_classes = [
    '杆塔缺螺栓',
'塔材变形',
'塔身异物',
'塔材锈蚀',
'塔身锈蚀'
#     ['缺销','缺垫片','螺母安装不规范','销钉安装不到位','螺母锈蚀',
#  '垫片锈蚀','螺帽锈蚀','缺螺母','螺丝安装不规范','缺螺栓','螺栓锈蚀','销钉锈蚀'],

 ]
sub_data_name = ['小金具']
data_roots = ['/data/wulianjun/StateGridv5/{}/Annotations'.format(x) for x in sub_data_name]

txt_roots = [os.path.dirname(data_root)+'/ImageSets/Main' for data_root in data_roots]

for i,(data_root,txt_root) in enumerate(zip(data_roots,txt_roots)):
    all_files = []
    for root, dirs, files in os.walk(data_root, topdown=False):
        all_files.append(files)
    # from pdb import set_trace
    # set_trace()
    for j in tqdm(range(len(all_files[0]))):
        file = all_files[0][j]
        xml_path = os.path.join(data_root,file)
        save_xml = read_xml(xml_path,dataset_train_classes[i])

from pdb import set_trace
set_trace()