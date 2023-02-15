# coding=utf-8
from multiprocessing.spawn import import_main_path
import os
import cv2
import numpy as np

import xml.etree.ElementTree as ET

Crop_size = (1333, 800)

keep_radio = True

path_root = '/home/public/dataset/stateGridv4/huakeyangben/'
dataset_names = ['daodixian','fushusheshi','ganta','jiedizhuangzhi','jichu',]
paths =[path_root+'{}/'.format(dataset_name) for dataset_name in dataset_names]
xml_path_roots = [path+'xml/' for path in paths]

target_paths = ["/home/lianjun.wu/wulianjun/{}_crop2/JPEGImages".format(dataset_name) for dataset_name in dataset_names]
target_xml_paths = ["/home/lianjun.wu/wulianjun/{}_crop2/Annotations".format(dataset_name) for dataset_name in dataset_names]



def read_xml(xml_path):
    bboxes = []
    DomTree = ET.parse(xml_path)
    root = DomTree.getroot()

    size = root.find('size')

    width = int(size.find('width').text)
    height = int(size.find('height').text)
    for objects in root.findall('object'):
        bnd_box = objects.find('bndbox')
        bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]
        bboxes.append(np.array(bbox))
    bboxes = np.array(bboxes)
    return width, height, bboxes

def resize_xml(xml_path,index, new_wight, new_height, crop_region,bbox_group):
    DomTree = ET.parse(xml_path)
    root = DomTree.getroot()
    size = root.find('size')
    filename = root.find('filename').text

    size.find('width').text = str(int(new_wight))
    size.find('height').text = str(int(new_height))
    root.find('filename').text = '{}_{}.{}'.format(filename.split('.')[0],str(index),filename.split('.')[1])
    
    for objects in root.findall('object'):
        bnd_box = objects.find('bndbox')
        bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]
        if bbox in bbox_group:
            bnd_box.find('xmin').text = str(bbox[0] - crop_region[0])
            bnd_box.find('ymin').text = str(bbox[1] - crop_region[1])
            bnd_box.find('xmax').text = str(bbox[2] - crop_region[0])
            bnd_box.find('ymax').text = str(bbox[3] - crop_region[1])
        else:
            root.remove(objects)
    
    write_dir = os.path.join(target_xml_path, '{}_{}.xml'.format(os.path.basename(xml_path).split('.')[0],index))
    DomTree.write(write_dir,encoding='utf-8')

def min_surround(bb, c_s):
    if len(bb) == 0:
        return np.array([False])
    Min_bbox_surround = []
    for bbox in bb:
        for i in range(len(bb)):
            Min_bbox_surround.append([
                min(bbox[0], bb[i][0]),
                min(bbox[1], bb[i][1]),
                max(bbox[2], bb[i][2]),
                max(bbox[3], bb[i][3])
            ])
    min_surrand_bbox = np.array(Min_bbox_surround)
    min_surrand_bbox_size = np.array([
        (min_surrand_bbox[:, 2] - min_surrand_bbox[:, 0]),
        (min_surrand_bbox[:, 3] - min_surrand_bbox[:, 1])
    ]).transpose(1, 0).reshape(len(bb), len(bb), 2)
    c_s = np.array(c_s).reshape(1, 1, 2).repeat(len(min_surrand_bbox_size),
                                                axis=0).repeat(
                                                    len(min_surrand_bbox_size),
                                                    axis=1)
    flag = min_surrand_bbox_size > c_s
    return flag

def spilt_group(bbox_group, insides_bboxes, C_s):
    outsides_bboxes = []
    while (True):
        flag = min_surround(insides_bboxes, C_s)
        if len(np.where(flag)[0]) == 0:
            bbox_group.append(insides_bboxes)

            break
        else:
            indx = np.argmax(np.bincount(np.where(flag)[0]))
            outsides_bboxes.append(insides_bboxes[indx])
            insides_bboxes = np.delete(insides_bboxes, indx, 0)
            if len(insides_bboxes) == 0:
                break
    return outsides_bboxes


def crop2(width, height, bboxes,crop_size):

    if width < crop_size[0]:
        C_s = (width, int(crop_size[1] * width / crop_size[0]))
    elif height < crop_size[1]:
        C_s = (int(crop_size[0] * height / crop_size[1]), height)
    else:
        C_s = crop_size

    if C_s[0]> width or C_s[1]> height:
        C_s = (min(C_s[0],width),min(C_s[0],height))
        
    # C_s = crop_size
    # if width < crop_size[0]:
    #     C_s[0] = width
    # if height < crop_size[1]:
    #     C_s[1] = height
    bbox_group = []
    outsides_bboxes = []
    insides_bboxes = bboxes

    while (True):
        outsides_bboxes_lastlen = len(outsides_bboxes)
        outsides_bboxes = spilt_group(bbox_group, insides_bboxes, C_s)
        if len(outsides_bboxes) == 0:
            break
        elif outsides_bboxes_lastlen == len(outsides_bboxes):

            for k in range(len(outsides_bboxes)):
                bbox_group.append(outsides_bboxes[k])
            break

        else:
            insides_bboxes = np.array(outsides_bboxes)
    crop_areas = []
    valid_indx = []
    for i,b_g in enumerate(bbox_group):
        if len(b_g.shape) == 1:
            b_g = b_g.reshape(1, -1)
        min_s = [
            min(b_g[:, 0]),
            min(b_g[:, 1]),
            max(b_g[:, 2]),
            max(b_g[:, 3])
        ]
        if min_s[3] - min_s[1] > C_s[1] or min_s[2] - min_s[0] > C_s[0]:
            continue

        else:
            valid_indx.append(i)
            min_s_center = [(min_s[0] + min_s[2]) / 2,
                            (min_s[3] + min_s[1]) / 2]
            crop_area = [
                int(min_s_center[0] - C_s[0] / 2),
                int(min_s_center[1] - C_s[1] / 2),
                int(min_s_center[0] + C_s[0] / 2),
                int(min_s_center[1] + C_s[1] / 2)
            ]
            crop_area = np.array(crop_area).astype(np.int32)
            delta = crop_area - np.array([0, 0, width, height])
            sig = np.array([1, 1, -1, -1])
            invalid = (sig * delta) < 0
            if len(np.where(invalid)[0]) != 0:
                invalid_1 = invalid[[2, 3, 0, 1]]
                invalid_2 = invalid | invalid_1
                if len(delta[::2][invalid[::2]]) != 0:
                    crop_area[::
                                2] = crop_area[::2] - invalid_2[::2].astype(
                                    np.int32) * delta[::2][invalid[::2]]
                if len(delta[1::2][invalid[1::2]]) != 0:
                    crop_area[1::2] = crop_area[1::2] - invalid_2[
                        1::2].astype(np.int32) * delta[1::2][invalid[1::2]]
            if crop_area[3] - crop_area[1] != C_s[1]:
                crop_area[3] = crop_area[1] + C_s[1]
            if crop_area[2] - crop_area[0] != C_s[0]:
                crop_area[2] = crop_area[0] + C_s[0]
            crop_areas.append(crop_area.astype(np.float32).reshape(1,-1))
    

    return crop_areas,bbox_group

for path,xml_path_root,target_xml_path,target_path in zip(paths,xml_path_roots,target_xml_paths,target_paths):

    if not os.path.exists(target_xml_path):
        if not os.path.exists(os.path.dirname(target_xml_path)):
            os.mkdir(os.path.dirname(target_xml_path))
        os.mkdir(target_xml_path)
    if not os.path.exists(target_path):
        if not os.path.exists(os.path.dirname(target_path)):
            os.mkdir(os.path.dirname(target_path))
        os.mkdir(target_path)

    files = os.listdir(path)
    j = 0
    for file in files:
        print(j)
        j += 1

        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            continue
        else:
            ext = os.path.splitext(file)
            if ext[1] != '.jpg' and ext[1] != '.JPG':
                continue
            xml_path = os.path.join(xml_path_root, ext[0] + '.xml')
            width, height, bboxes = read_xml(xml_path)
            
            # if width>1333 and height>800:
            #     continue

            if width < Crop_size[0]:
                C_s = (width, int(Crop_size[1] * width / Crop_size[0]))
            elif height < Crop_size[1]:
                C_s = (int(Crop_size[0] * height / Crop_size[1]), height)
            else:
                C_s = Crop_size

            if C_s[0]> width or C_s[1]> height:

                # C_s = (min(C_s[0],width),min(C_s[0],height))
            
            # aa = os.path.join(target_path,'{}_{}.jpg'.format(file.split('.')[0],0))
            
                crop_areas,bbox_groups = crop2(width, height, bboxes,Crop_size)
                img = cv2.imread(file_path)
                

                for i,(crop_area,bbox_group) in enumerate(zip(crop_areas,bbox_groups)):
                    crop_area = crop_area.astype(np.int32).tolist()[0]
                    new_width = crop_area[2] - crop_area[0]
                    new_height = crop_area[3] - crop_area[1]
                    img_crop = img[crop_area[1]:crop_area[3],crop_area[0]:crop_area[2],:]
                    resize_xml(xml_path,i,new_width,new_height,crop_area,bbox_group.tolist())
                    new_img_name = '{}_{}.jpg'.format(file.split('.')[0],i)
                    cv2.imwrite(os.path.join(target_path,new_img_name),img_crop)
                # cv2.imwrite(os.path.join(target_path,new_img_name),img)

            
            

            

