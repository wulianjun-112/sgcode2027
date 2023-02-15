# coding=utf-8
from re import S
from PIL import Image, ImageDraw, ImageFont
import mmcv
import cv2
import numpy as np 
import xml.etree.ElementTree as ET
import os
# CLASSES = ['060800021','060800031','060800011','060800023','060800033','060800013']

# CLASSES = ['020001011', '020000031', '020001031'
# , '020100031', '020000011', '020000111', '020000021', '020001061'
# , '020100051', '020100011', '020001021', '020100021']

# label_name = ['引流线断股','导线松股','引流线松股','地线锈蚀','导线断股','导线异物','导线损伤','引流线异物','地线异物','地线断股','引流线损伤','地线损伤']
# xml_root = '/home/public/dataset/stateGridv4/daodixian/Annotations/'
# img_root = '/home/public/dataset/stateGridv4/daodixian/JPEGImages/'


# CLASSES  = ['020000111','020001011', '020000031' ,'020001031' ,
# '020100031', '020000011','020000021']
# label_name = ['导线异物','引流线断股','导线松股','引流线松股','地线锈蚀','导线断股','导线损伤']
# CLASSES = ['060800021','060800031','060800011']
# label_name = ['推土机','挖掘机','塔吊']
# xml_root = '/home/lianjun.wu/wulianjun/daodixian_crop2/Annotations/'
# img_root = '/home/lianjun.wu/wulianjun/daodixian_crop2/JPEGImages/'

CLASSES = ['线松股','线异物','线断股','线损伤']
train_class = ['线松股','线异物','线断股','线损伤']
val_classes = ['线松股','线异物','线断股','线损伤']
label_name = ['线松股','线异物','线断股','线损伤']
xml_root = '/data/wulianjun/datasets/SG5/导地线/Annotations_crop_no_fliter/'
img_root = '//data/wulianjun/datasets/SG5/导地线/JPEGImages_crop_no_fliter/'
def cv2ImgAddText(img, texts, bboxes,img_path, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)): # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
# 字体的格式
        fontStyle = ImageFont.truetype(
            '/data/wulianjun/code/simsun.ttf', textSize, encoding="utf-8")
# 绘制文本
        for text,(left, top) in zip(texts,bboxes):
            draw.text((left, top-20), text, textColor, font=fontStyle)
        img.save('/data/wulianjun/code/daodixian_vis/'+img_path)


def readXml(xmlfile,img_path):
    
    
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
            bboxes.append(bbox)
    from pdb import set_trace
    set_trace()        
    if len(bboxes)==0:
        return None
    # if xmlfile == '#60塔左侧地线横担鸟巢-其他-20170920-霍永良_2.xml':
    #     from pdb import set_trace
    #     set_trace()
    img = mmcv.imread(img_root+img_path).astype(np.uint8)  
    for bbox in bboxes:
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),(0,0,255), 4)    
    
    cv2ImgAddText(img,texts,[bbox[:2] for bbox in bboxes],img_path)
    




# from pdb import set_trace
# set_trace()
# f = open('/home/public/dataset/stateGridv4/TestSets/tongdao/test_6class.txt','r')
# all_xml_path = []
# all_img_path = []
# for line in f:
#     all_xml_path.append(line.split('\n')[0]+'.xml')
#     all_img_path.append(line.split('\n')[0])
# from pdb import set_trace
# set_trace()
xml_filenames = os.listdir(xml_root)
img_filenames = os.listdir(img_root)
xml_filenames.sort()
img_filenames.sort()
i = 0

for xml_path,img_path in zip(xml_filenames,img_filenames):
    if i % 20 ==0:
        print(i)
    i += 1
    # from pdb import set_trace
    # set_trace()
    # pass
    readXml(xml_path,img_path)
    





