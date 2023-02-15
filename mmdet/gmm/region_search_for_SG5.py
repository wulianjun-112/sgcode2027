from mmdet.gmm.gmm import GaussianMixture
import math
import cv2
import os
import os.path as osp
import numpy as np
from scipy.stats import percentileofscore
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pycocotools.coco import COCO
import torch
train_classes = ['线异物'
,'线断股'
,'线松股'
,'线损伤']

def get_points(image_size,
                every_dim_point_num,
                flatten=False):
    """Get points of a single scale level."""
    w, h = image_size
    # First create Range with the default dtype, than convert to
    # target `dtype` for onnx exporting.
    x_range = (torch.arange(every_dim_point_num)*(w//every_dim_point_num)).float()
    y_range = (torch.arange(every_dim_point_num)*(h//every_dim_point_num)).float()
    y, x = torch.meshgrid(y_range, x_range)
    if flatten:
        y = y.flatten()
        x = x.flatten()
    return y, x

def draw_img_with_gt(path,boxes):
    img = cv2.imread(path)
    for box in boxes:
        box = np.array(box,dtype=np.int32).tolist()
        img = cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,0,255))
    cv2.imwrite('./1.jpg',img)


anno_dir = '/data/wulianjun/datasets/SG5/导地线/ImageSets/Main/train_split.txt'
train_cat2label =  {cat: i for i, cat in enumerate(train_classes)}

train_data_infos = []
train_all_bboxes = []
train_all_whs = []
train_all_img_whs = []
train_all_labels = []

with open(anno_dir,'r',encoding='UTF-8') as f:
    fileids = f.readlines()

    dirname = os.path.dirname(os.path.dirname(os.path.dirname(anno_dir)))
    for fileid in fileids:
        bboxes = []
        anno_file = os.path.join(dirname, "Annotations", fileid.split('\n')[0] + ".xml")
        tree = ET.parse(anno_file)
        img_w,img_h = int(tree.getroot().find('size').find('width').text),int(tree.getroot().find('size').find('height').text)
        # if resize_keep_ratio[0] / img_w * img_h <= 800:
        #     scale_ratio = resize_keep_ratio[0] / img_w
        # else:
        #     scale_ratio = resize_keep_ratio[1] / img_h
        once = True
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if cls not in train_classes:
                continue
            if once:
                train_all_img_whs.append((img_w,img_h))
                once=False
            train_all_labels.append(train_cat2label[cls])
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            W,H = bbox[2] - bbox[0] , bbox[3] - bbox[1]
            
            if W>0 and H>0 :
                # train_all_whs.append((W*scale_ratio,H*scale_ratio))
                train_all_whs.append((W,H))
                bboxes.append(bbox)
        train_all_bboxes.append(bboxes)
        
        if not once:
            num_point = 16 
            ref_point = get_points((img_w,img_h),every_dim_point_num=num_point)

            #(H,W)
            ref_point = torch.stack(ref_point,dim=-1).reshape(-1,2)
            bboxes_tensor = torch.tensor(bboxes)
            center = torch.vstack([(bboxes_tensor[:,3]+bboxes_tensor[:,1])/2,(bboxes_tensor[:,0]+bboxes_tensor[:,2])/2]).T
            
            num_gt = len(bboxes_tensor)
            num_cluster = int(math.log2(num_gt) + 1)
            x = (ref_point[None,:,:] - center[:,None,:]) / torch.tensor([img_h,img_w]).reshape(1,1,2)
            x = x.flatten(1)
            
            
            # if num_gt == 1:

            # draw_img_with_gt(anno_file.replace('Annotations','JPEGImages').replace('.xml','.jpg'),bboxes_tensor)
            if num_gt == 1:
                offsets = torch.randn((1,2))
                
                continue
            
            from pdb import set_trace
            set_trace()
            gmm = GaussianMixture(num_cluster,num_point*num_point*2,covariance_type='diag')
            gmm.fit(x)
            cluster_index = gmm.predict(x)
            cluster_mean = gmm.mu.mean(-1).squeeze(0)
            cluster_var = gmm.var.mean(-1).squeeze(0)
            torch.randn((num_cluster,2))
    # from pdb import set_trace
    # set_trace()