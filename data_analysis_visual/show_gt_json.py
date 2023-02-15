# coding=utf-8
from re import S
from PIL import Image, ImageDraw, ImageFont
import mmcv
import cv2
import numpy as np 
from pycocotools.coco import COCO
import json
import os
# CLASSES = ['060800021','060800031','060800011','060800023','060800033','060800013']
# CLASSES = ['020001011', '020000031', '020001031'
# , '020100031', '020000011', '020000111', '020000021', '020001061'
# , '020100051', '020100011', '020001021', '020100021']

# label_name = ['引流线断股','导线松股','引流线松股','地线锈蚀','导线断股','导线异物','导线损伤','引流线异物','地线异物','地线断股','引流线损伤','地线损伤']
# xml_root = '/home/public/dataset/stateGridv4/daodixian/Annotations/'
# img_root = '/home/public/dataset/stateGridv4/daodixian/JPEGImages/'
# CLASSES = ['060800021','060800031','060800011']
label_name = ['推土机','挖掘机','塔吊']
json_root = '/home/lianjun.wu/wulianjun/Co-mining/fcos.res50.cocomiss50/inference/coco_instances_results.json'
img_root = '/home/public/dataset/COCO/coco/test2017'

def cv2ImgAddText(img, texts, bboxes,img_path, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)): # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
# 字体的格式
        fontStyle = ImageFont.truetype(
            '/home/lianjun.wu/wulianjun/msyh.ttf', textSize, encoding="utf-8")
# 绘制文本
        for text,(left, top) in zip(texts,bboxes):
            draw.text((left, top-20), text, textColor, font=fontStyle)
        img.save('/home/lianjun.wu/wulianjun/mmdetection/tongdao_crop2_showgt/'+img_path)


def readXml(xmlfile,img_path):
    
    img = mmcv.imread(img_root+img_path).astype(np.uint8)
    DomTree = ET.parse(xml_root+xmlfile)

    root = DomTree.getroot()
    bboxes = []
    texts = []
    for objects in root.findall('object'):
        class_label = objects.find('name').text
        if class_label in CLASSES :
            # if class_label in classNames:
            bnd_box = objects.find('bndbox')
            bbox = [
                        int(float(bnd_box.find('xmin').text)),
                        int(float(bnd_box.find('ymin').text)),
                        int(float(bnd_box.find('xmax').text)),
                        int(float(bnd_box.find('ymax').text))
                    ]
            texts.append(label_name[CLASSES.index(class_label)])
            bboxes.append(bbox[:2])
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),(0,0,255), 4)    
        
    cv2ImgAddText(img,texts,bboxes,img_path)
    




from pdb import set_trace
set_trace()
dataset = COCO(json_root)
img_ids = dataset.getImgIds()

data_infos =[]
            
for i in img_ids:
    info = dataset.loadImgs([i])[0]
    data_infos.append(info)




