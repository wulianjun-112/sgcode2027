import argparse
import imp
import json
from multiprocessing.spawn import import_main_path
import os
import os.path as osp
import numpy as np
from scipy.stats import percentileofscore
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pycocotools.coco import COCO
import megfile
import torch
train_classes = ['020000111','020001011', '020000031' ,'020001031' ,
'020100031', '020000011','020000021']

val_classes = ['020000111' , '020001013' ,'020000031' ,'020001031', 
'020100032','020000013', '020000023' ,'020000113']

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')

    parser.add_argument(
        '--anno-dir',
        type=str,
        help='path of annotations file')
        
    
    parser.add_argument(
        '--test-anno-dir',
        type=str,
        help='path of test annotations file')

    parser.add_argument(
        '--dirname',
        type=str,
        help='path of jpg and xml for voc type datasets')

    parser.add_argument(
        '--output-name',
        type=str,
        help='path of jpg and xml for voc type datasets')

    args = parser.parse_args()
    return args

args = parse_args()
train_cat2label =  {cat: i for i, cat in enumerate(train_classes)}
val_cat2label =  {cat: i for i, cat in enumerate(val_classes)}

assert args.anno_dir is not None
train_data_infos = []
train_all_bboxes = []
train_all_labels = []
assert osp.isfile(args.anno_dir) 
if  args.anno_dir.endswith('json'):
    dataset = COCO(args.anno_dir)
    img_ids = dataset.getImgIds()
    cats = dataset.loadCats(dataset.getCatIds())
    label_ids={cat['id']: i for i, cat in enumerate(cats)}
    for i in img_ids:
        info = dataset.loadImgs([i])[0]
        train_data_infos.append(info)

        ann_ids = dataset.getAnnIds(imgIds=[i])
        bboxes = np.array([ann['bbox'] for ann in dataset.loadAnns(ann_ids)]) 
        labels = np.array([label_ids[ann['category_id']] for ann in  dataset.loadAnns(ann_ids)])
        train_all_bboxes.append(bboxes)
        train_all_labels.append(labels)
elif args.anno_dir.endswith('txt'):
    with megfile.smart_open(args.anno_dir) as f:
        train_fileids = np.loadtxt(f, dtype=np.str)
        dirname = os.path.dirname(os.path.dirname(os.path.dirname(args.anno_dir)))
        for fileid in train_fileids:
            train_labels = []
            anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
            tree = ET.parse(anno_file)
            for obj in tree.findall("object"):
                
                cls = obj.find("name").text
                if cls not in train_classes:
                    continue
                train_labels.append(train_cat2label[cls])
                bbox = obj.find("bndbox")
                bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
                bbox[0] -= 1.0
                bbox[1] -= 1.0

                train_all_bboxes.append(bbox)
            train_all_labels.append(train_labels)
else:
    raise NotImplementedError


if args.test_anno_dir is not None:
    test_data_infos = []
    test_all_bboxes = []
    test_all_labels = []
    assert osp.isfile(args.test_anno_dir) 
    if args.test_anno_dir.endswith('json'):
        dataset = COCO(args.test_anno_dir)
        img_ids = dataset.getImgIds()
        cats = dataset.loadCats(dataset.getCatIds())
        label_ids={cat['id']: i for i, cat in enumerate(cats)}
        for i in img_ids:
            info = dataset.loadImgs([i])[0]
            test_data_infos.append(info)

            ann_ids = dataset.getAnnIds(imgIds=[i])
            bboxes = np.array([ann['bbox'] for ann in dataset.loadAnns(ann_ids)]) 
            labels = np.array([label_ids[ann['category_id']] for ann in  dataset.loadAnns(ann_ids)])
            test_all_bboxes.append(bboxes)
            test_all_labels.append(labels)
    elif args.test_anno_dir.endswith('txt'):
        with megfile.smart_open(args.test_anno_dir) as f:
            fileids = np.loadtxt(f, dtype=np.str)
            dirname = os.path.dirname(os.path.dirname(os.path.dirname(args.test_anno_dir)))
        for fileid in fileids:
            bboxes = []
            labels = []
            anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
            tree = ET.parse(anno_file)
            for obj in tree.findall("object"):
                cls = obj.find("name").text
                if cls not in val_classes:
                    continue
                labels.append(val_cat2label[cls] % len(train_classes))
                bbox = obj.find("bndbox")
                bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
                bbox[0] -= 1.0
                bbox[1] -= 1.0

                bboxes.append(bbox)
            test_all_labels.append(labels)
    else:
        raise NotImplementedError


np_all_labels = np.zeros((len(train_all_labels),len(train_classes)))
for i,train_labels in enumerate(train_all_labels):
    for train_label in train_labels:
        np_all_labels[i,train_label] += 1

train_all_labels = torch.tensor(np_all_labels)
train_radio = train_all_labels.sum(dim=0)/train_all_labels.sum()

np_test_all_labels = np.zeros((len(test_all_labels),len(train_classes)))
for i,test_labels in enumerate(test_all_labels):
    for test_label in test_labels:
        np_test_all_labels[i,test_label] += 1

val_all_labels = torch.tensor(np_test_all_labels)
val_radio = val_all_labels.sum(dim=0)/val_all_labels.sum()
from pdb import set_trace
set_trace()

val_radio_ = val_radio.clone()
 


max_radio,max_idx = val_radio_.max(dim=0)

temp = train_all_labels[torch.where(train_all_labels[:,max_idx])[0],:]
train_labels_ = temp.sum(dim=0)[0]/max_radio * val_radio_
inter_train_label = train_labels_ - temp.sum(dim=0)
inter_train_label = inter_train_label.int()

aa = np.array(torch.where(train_all_labels[:,4])[0])
np.random.shuffle(aa)

inds_4 = torch.tensor(aa)
sum_ = 0
i_ = 0
for ind_4 in inds_4:
    sum_ += train_all_labels[ind_4,4]
    if sum_ >= inter_train_label[4]:
        break
    i_ +=1 
inds_4 = inds_4[:i_]

inter_train_label = inter_train_label - train_all_labels[inds_4,:].sum(dim=0).int()

inds_0 = torch.where(train_all_labels[:,0])[0]
inds_5 = torch.where(train_all_labels[:,5])[0]
inds_3 = torch.where(train_all_labels[:,3])[0]
aa = np.array(torch.where(train_all_labels[:,3])[0])
np.random.shuffle(aa)
sum_ = 0
i_ = 0
inds_3 = torch.tensor(aa)
for ind_3 in inds_3:
    sum_ += train_all_labels[ind_3,3]
    if sum_ >= inter_train_label[3]:
        break
    i_ +=1 
inds_3 = inds_3[:i_]
from pdb import set_trace
set_trace()
train_all_labels[torch.cat((inds_0,inds_4,inds_3,inds_5)).unique(),:].sum(0)
new_train_fileids = train_fileids[np.array(torch.cat((inds_0,inds_4,inds_3,inds_5)).unique())]

f = open('./trainval.txt','w')
for fileid in new_train_fileids:
    f.write(fileid+'\n')
f.close()
# loss = torch.abs(val_radio-train_radio).sum()

# all_index = torch.randperm(len(train_all_labels))

# remain_idx = []
# for index in all_index:
#     temp = train_all_labels[index].clone()
#     train_all_labels[index] = 0.
#     train_radio = train_all_labels.sum(dim=0)/train_all_labels.sum()
#     new_loss = torch.abs(val_radio-train_radio).sum()
#     if new_loss>=loss:
#         train_all_labels[index] = temp.clone()
#     else:
#         remain_idx.append(index)
# from pdb import set_trace
# set_trace()

# train_all_labels = np.array(train_all_labels)
