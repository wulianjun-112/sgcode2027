#coding=utf-8
import os
import sys
import io
from matplotlib import scale
# from mmdetection.mmdet.core import bbox
import numpy as np
import datetime

import matplotlib.pyplot as plt
# np.set_printoptions(suppress=True)
import xml.etree.ElementTree as ET
import xml.dom.minidom
from xml.dom.minidom import Document
import multiprocessing as mp


difficult_obj = dict()

width,height = 0 , 0

def create_detail_day():
    '''
 :return:
 '''
    daytime = datetime.datetime.now().strftime('day' + '%Y_%m_%d')
    detail_time = daytime
    return detail_time


def make_print_to_file(filename, path='./'):
    '''
  example:
 use make_print_to_file() , and the all the information of funtion print , will be write in to a log file
 :param path: the path to save print information
 :return:
 '''
    class Logger(object):
        def __init__(self, filename=filename, path="./"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    sys.stdout = Logger(filename + create_detail_day() + '.log', path=path)



def print_statistics():

    print('*' * 30)
    for i, key in enumerate(dict_label):
        print(i, ':', key)
    print('*' * 30)
    print('*' * 30)
    print('各类别标注框数量：', dict_label)
    print('*' * 30)
    class_name = []
    class_num = []

    for key, item in dict_label.items():

        class_num.append(item)
    per_pic_gt_num = []
    for key, item in dict_img.items():
        per_pic_gt_num.append(item)
    print('*' * 30)
    print('标注框总和：', sum(class_num))
    print('*' * 30)
    print('数据集图片数：', len(per_pic_gt_num))
    print('*' * 30)
    print('每张图片的标注框数量均值：', np.mean(per_pic_gt_num))
    print('每张图片的标注框数量方差：', np.std(per_pic_gt_num))
    print('*' * 30)
    # print('类别标注框数量方差：',np.std(class_num))

    scale_bbox = dict()
    for key, item in dict_bbox.items():
        scale_bbox[key] = [
            np.sqrt((box[2] - box[0]) * (box[3] - box[1])) for box in item
        ]
    num_LT_32 = 0
    num_LT_256_GT_32 = 0
    num_GT_256 = 0
    print('*' * 30)
    for key, item in scale_bbox.items():
        
        # item = np.array(item)
        print(key + '类标注框数量', len(item))
        try:
            print(key+'类标注框平均尺寸sqrt',int(np.mean(item)))
        except:
            from pdb import set_trace
            set_trace()
                

        print(key+'类标注框最大尺寸sqrt',int(max(item)))
        print(key+'类标注框最小尺寸sqrt',int(min(item)))
        # print('*'*30)
        np_item = np.array(item)
        print(key+'类标注框尺寸<32:',len(np.where(np_item<32)[0]))
        print(key+' 32<scale<128:',len(np.where(np_item<128)[0])-len(np.where(np_item<32)[0]))
        print(key+' >128:',len(np.where(np_item>128)[0]))
        # print('*'*30)
        num_LT_32 += len(np.where(np_item<32)[0])
        num_LT_256_GT_32 += len(np.where(np_item<128)[0])-len(np.where(np_item<32)[0])
        num_GT_256 += len(np.where(np_item>128)[0])

        
    print('*'*30)
    print('尺寸小于32标注框总数',num_LT_32)
    print('尺寸大于32小于256标注框总数',num_LT_256_GT_32)
    print('尺寸大于256标注框总数',num_GT_256)
    print('*' * 30)

    # fig = plt.figure()
    # plt.bar(range(len(class_num)), class_num,width=0.5,tick_label=class_name,linewidth = 500)
    # fig.savefig('per_class_num.png')


def readXml(xmlfile,dict_label,dict_img,dict_bbox):

    DomTree = ET.parse(xmlfile)

    root = DomTree.getroot()
    size = root.find('size')
    if size is not None:
        width = int(size.find('width').text)
        height = int(size.find('height').text)
    # objectlist = annotation.getElementsByTagName('object')

    for objects in root.findall('object'):
        difficult = objects.find('difficult').text

        if difficult == '1':
            if xmlfile not in difficult_obj.keys():
                difficult_obj[xmlfile] = 1
            else:
                difficult_obj[xmlfile] += 1
        else:
            class_label = objects.find('name').text

            # if class_label in classNames:
            if xmlfile not in dict_img.keys():
                dict_img[xmlfile] = 1
            else:
                dict_img[xmlfile] += 1
            if class_label not in dict_label.keys():
                dict_label[class_label] = 1
            else:
                dict_label[class_label] += 1
            bnd_box = objects.find('bndbox')

            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3] or min(bbox)<0:
                continue
            else:
                if class_label not in dict_bbox.keys():
                    dict_bbox[class_label] = []
                dict_bbox[class_label].append(bbox)


# data_brances = ['导地线','附属设施','杆塔','基础','绝缘子','通道环境','小金具','大金具', '接地装置']
data_brances = ['新验证集']
data_root = '/data/wulianjun/datasets/SG5'
# data_brances = ['Valset']
# data_root = '/home/public/dataset/stateGridv4/'
# train_xml_path = []
# val_xml_path = []
all_files = []
data_roots = [
    os.path.join(data_root, data_brance,'Annotations') for data_brance in data_brances
]
# data_roots = ['/home/public/dataset/stateGridv4/ValSet/Annotations']
for data_root in data_roots:
    for root, dirs, files in os.walk(data_root, topdown=False):
        all_files.append(files)


aa = 0
make_print_to_file('SG5_valset_analysis_')

for q,root, files in zip(range(len(data_brances)),data_roots, all_files):
    train_xml_path = []
    aa = 0
    dict_label = dict()
    dict_img = dict()
    dict_bbox = dict()
    
    for file in files:
        train_xml_path.append(os.path.join(root, file))

    for xml_path in train_xml_path:
        # if aa % 100 == 0:
            # print(aa)
        # aa += 1
        # if aa ==2506:
        # from pdb import set_trace
        # set_trace()
        readXml(xml_path,dict_label,dict_img,dict_bbox)  
    # from pdb import set_trace
    # set_trace()
    print('*' * 30, 'Train_Set', '*' * 30)
    print(data_brances[q])
    print_statistics()
    print('*' * 70)






# print(data_root)


# print('*' * 30, 'Train_Set', '*' * 30)
# print_statistics()
# print('*' * 70)
# from pdb import set_trace
# set_trace()
# from pdb import set_trace
# set_trace()
# dict_label = dict()
# dict_img = dict()
# difficult_obj = dict()
# dict_bbox = dict()

# for xml_path in all_val_xml_path:
#     readXml(xml_path)

# print('*' * 30, 'Val_Set', '*' * 30)
# print_statistics()
# print('*' * 70)
